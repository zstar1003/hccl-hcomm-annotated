/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ut_dispatch.h"
#include "ra_hdc_tlv.h"
#include "rs_tlv.h"
#include "ra_adp_tlv.h"
#include "ra_rs_err.h"
#include <stdlib.h>
#include "securec.h"

#define TC_TLV_HDC_MSG_SIZE (32 * 1024)

void TcRaRsTlvInit()
{
    union OpTlvInitData dataIn;
    union OpTlvInitData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvInitData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvInitData));

    dataIn.txData.phyId = 0;
    mocker((stub_fn_t)RsTlvInit, 1, 0);
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpTlvInitData), &dataIn, sizeof(union OpTlvInitData));
    ret = RaRsTlvInit(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsTlvInit, 1, -1);
    ret = RaRsTlvInit(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsTlvInit, 1, -ENOTSUPP);
    ret = RaRsTlvInit(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaRsTlvDeinit()
{
    union OpTlvDeinitData dataIn;
    union OpTlvDeinitData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvDeinitData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvDeinitData));

    dataIn.txData.phyId = 0;
    mocker((stub_fn_t)RsTlvDeinit, 1, 0);
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpTlvDeinitData), &dataIn, sizeof(union OpTlvDeinitData));
    ret = RaRsTlvDeinit(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsTlvDeinit, 1, -1);
    ret = RaRsTlvDeinit(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaRsTlvRequest()
{
    union OpTlvRequestData dataIn;
    union OpTlvRequestData dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvRequestData));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvRequestData));

    dataIn.txData.head.moduleType = TLV_MODULE_TYPE_NSLB;
    dataIn.txData.head.phyId = 0;
    mocker((stub_fn_t)RsTlvRequest, 1, 0);
    memcpy_s(inBuf + sizeof(struct MsgHead), sizeof(union OpTlvRequestData), &dataIn, sizeof(union OpTlvRequestData));
    ret = RaRsTlvRequest(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsTlvRequest, 1, -1);
    ret = RaRsTlvRequest(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

void TcRaRsTlvRequestV2()
{
    union OpTlvRequestDataV2 dataIn;
    union OpTlvRequestDataV2 dataOut;
    int rcvBufLen = 0;
    int opResult = 0;
    int outLen = 0;
    int ret;

    char* inBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvRequestDataV2));
    char* outBuf = calloc(1, sizeof(struct MsgHead) + sizeof(union OpTlvRequestDataV2));

    dataIn.txData.head.moduleType = TLV_MODULE_TYPE_CCU;
    dataIn.txData.head.phyId = 0;
    mocker((stub_fn_t)RsTlvRequest, 1, 0);
    memcpy_s(
        inBuf + sizeof(struct MsgHead), sizeof(union OpTlvRequestDataV2), &dataIn, sizeof(union OpTlvRequestDataV2));
    ret = RaRsTlvRequest(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker((stub_fn_t)RsTlvRequest, 1, -1);
    ret = RaRsTlvRequest(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(inBuf);
    inBuf = NULL;
    free(outBuf);
    outBuf = NULL;
}

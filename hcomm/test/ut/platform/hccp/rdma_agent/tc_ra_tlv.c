/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_tlv.h"
#include "securec.h"
#include "ut_dispatch.h"
#include "dl_hal_function.h"
#include "ra_tlv.h"
#include "ra_hdc_tlv.h"

#define TC_TLV_MSG_SIZE (64 * 1024)
#define TC_TLV_MSG_SIZE_INVALID (64 * 1024 + 1U)

void TcRaTlvInit()
{
    struct TlvInitInfo initInfo = {0};
    struct RaTlvHandle* tlvHandleTmp = NULL;
    unsigned int bufferSize = 0;
    int ret = 0;

    initInfo.nicPosition = NETWORK_OFFLINE;
    initInfo.phyId = 0;

    mocker(memcpy_s, 10, 0);
    mocker(RaHdcTlvInit, 100, -1);
    ret = RaTlvInit(&initInfo, &bufferSize, (void**)&tlvHandleTmp);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcTlvInit, 100, 0);
    mocker(memcpy_s, 10, 0);
    mocker(pthread_mutex_init, 100, -1);
    ret = RaTlvInit(&initInfo, &bufferSize, (void**)&tlvHandleTmp);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcTlvInit, 100, 0);
    mocker(memcpy_s, 10, 0);
    mocker(pthread_mutex_init, 100, 0);
    ret = RaTlvInit(&initInfo, &bufferSize, (void**)&tlvHandleTmp);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
    free(tlvHandleTmp);
    tlvHandleTmp = NULL;
}

void TcRaTlvDeinit()
{
    struct RaTlvHandle* tlvHandleTmp = calloc(1, sizeof(struct RaTlvHandle));
    struct RaTlvOps tlvOps = {0};
    int ret = 0;

    tlvHandleTmp->tlvOps = &tlvOps;
    tlvOps.raTlvDeinit = NULL;
    tlvHandleTmp->initInfo.phyId = 0;
    ret = RaTlvDeinit(tlvHandleTmp);
    EXPECT_INT_NE(0, ret);

    tlvHandleTmp = calloc(1, sizeof(struct RaTlvHandle));
    tlvHandleTmp->tlvOps = &tlvOps;
    tlvHandleTmp->initInfo.phyId = 0;
    tlvOps.raTlvDeinit = RaHdcTlvDeinit;
    mocker(RaHdcTlvDeinit, 100, -1);
    ret = RaTlvDeinit(tlvHandleTmp);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    tlvHandleTmp = calloc(1, sizeof(struct RaTlvHandle));
    tlvHandleTmp->tlvOps = &tlvOps;
    tlvHandleTmp->initInfo.phyId = 0;
    tlvOps.raTlvDeinit = RaHdcTlvDeinit;
    mocker(RaHdcTlvDeinit, 100, 0);
    ret = RaTlvDeinit(tlvHandleTmp);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRaTlvRequest()
{
    unsigned int moduleType = TLV_MODULE_TYPE_CCU;
    struct RaTlvHandle tlvHandleTmp = {0};
    struct RaTlvOps tlvOps = {0};
    struct TlvMsg sendMsg = {0};
    struct TlvMsg recvMsg = {0};
    int ret = 0;

    tlvHandleTmp.bufferSize = TC_TLV_MSG_SIZE;
    sendMsg.length = TC_TLV_MSG_SIZE_INVALID;
    ret = RaTlvRequest(&tlvHandleTmp, moduleType, &sendMsg, &recvMsg);
    EXPECT_INT_NE(0, ret);

    tlvHandleTmp.tlvOps = &tlvOps;
    tlvOps.raTlvRequest = NULL;
    sendMsg.length = TC_TLV_MSG_SIZE;
    ret = RaTlvRequest(&tlvHandleTmp, moduleType, &sendMsg, &recvMsg);
    EXPECT_INT_NE(0, ret);

    tlvOps.raTlvRequest = RaHdcTlvRequest;
    mocker(RaHdcTlvRequest, 100, -1);
    ret = RaTlvRequest(&tlvHandleTmp, moduleType, &sendMsg, &recvMsg);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RaHdcTlvRequest, 100, 0);
    ret = RaTlvRequest(&tlvHandleTmp, moduleType, &sendMsg, &recvMsg);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

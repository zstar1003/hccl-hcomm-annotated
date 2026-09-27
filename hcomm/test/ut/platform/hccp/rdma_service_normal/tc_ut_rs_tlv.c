/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <pthread.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "dl_netco_function.h"
#include "dl_ccu_function.h"
#include "ut_dispatch.h"
#include "hccp_tlv.h"
#include "rs_inner.h"
#include "rs_tlv.h"
#include "rs_epoll.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "file_opt.h"
#include "rs_adp_nslb.h"
#include "network_comm.h"

static struct rs_cb stubRsCb;
extern int RsTlvAssembleSendData(
    struct TlvBufInfo* bufInfo, struct TlvRequestMsgHead* head, char* data, bool* sendFinish, int dataMaxLength);
extern void RsEpollEventHandleOne(struct rs_cb* rsCb, struct epoll_event* events);
extern int RsNslbRequest(struct TlvRequestMsgHead* head, char* data);
extern int RsNetcoTblApiInit(void);
extern int RsCcuRequest(struct TlvRequestMsgHead* head, char* dataIn, char* dataOut, unsigned int* bufferSize);
extern int RsGetTlvCb(uint32_t phyId, struct RsTlvCb** tlvCb);
extern int RsNetcoInitArg(unsigned int phyId, NetCoIpPortArg* netcoArg);
extern int RsNetcoApiInit(void);
extern int RsNslbNetcoInit(unsigned int phyId, struct RsNslbCb* nslbCb);
extern void RsNslbNetcoDeinit(struct RsNslbCb* nslbCb);

int StubRsGetNslbCb(uint32_t phyId, struct RsTlvCb** tlvCb)
{
    stubRsCb.connCb.epollfd = 0;
    stubRsCb.tlvCb.bufInfo.bufferSize = RS_TLV_BUFFER_SIZE;
    stubRsCb.tlvCb.bufInfo.buf = (char*)calloc(stubRsCb.tlvCb.bufInfo.bufferSize, sizeof(char));
    stubRsCb.tlvCb.initFlag = false;
    pthread_mutex_init(&stubRsCb.tlvCb.mutex, NULL);
    *tlvCb = &stubRsCb.tlvCb;
    return 0;
}

int StubRsGetNslbCbDeinit(uint32_t phyId, struct RsTlvCb** tlvCb)
{
    stubRsCb.connCb.epollfd = 0;
    stubRsCb.tlvCb.bufInfo.bufferSize = RS_TLV_BUFFER_SIZE;
    stubRsCb.tlvCb.bufInfo.buf = (char*)calloc(stubRsCb.tlvCb.bufInfo.bufferSize, sizeof(char));
    stubRsCb.tlvCb.initFlag = true;
    pthread_mutex_init(&stubRsCb.tlvCb.mutex, NULL);
    *tlvCb = &stubRsCb.tlvCb;
    return 0;
}

int StubRsGetNslbCbAfterDeinit(uint32_t phyId, struct RsTlvCb** tlvCb)
{
    *tlvCb = &stubRsCb.tlvCb;
    return 0;
}

int StubRsGetNslbCbInit(uint32_t phyId, struct RsTlvCb** tlvCb)
{
    stubRsCb.tlvCb.initFlag = false;
    *tlvCb = &stubRsCb.tlvCb;
    return 0;
}

int StubRsTlvAssembleSendData(
    struct TlvBufInfo* bufInfo, struct TlvRequestMsgHead* head, char* data, bool* sendFinish, int dataMaxLength)
{
    if (head->offset == 0) {
        *sendFinish = false;
    } else {
        *sendFinish = true;
    }
    return 0;
}

int StubRsGetRsCbV2(unsigned int phyId, struct rs_cb** rsCb)
{
    stubRsCb.tlvCb.initFlag = false;
    stubRsCb.connCb.epollfd = 0;
    *rsCb = &stubRsCb;
    return 0;
}

int StubFileReadCfg(const char* filePath, int devId, const char* confName, char* confValue, unsigned int len)
{
    if (strncmp(confName, "udp_port_mode", strlen("udp_port_mode") + 1) == 0) {
        memcpy_s(confValue, len, "nslb_dp", strlen("nslb_dp"));
    } else {
        memcpy_s(confValue, len, "16666", strlen("16666"));
    }
    return 0;
}

void FreeRsCb()
{
    pthread_mutex_destroy(&stubRsCb.tlvCb.mutex);
    free(stubRsCb.tlvCb.bufInfo.buf);
    stubRsCb.tlvCb.bufInfo.buf = NULL;
}

void TcRsNslbInit()
{
    unsigned int bufferSize = 0;
    unsigned int phyId = 0;
    int ret = 0;

    mocker_invoke(RsGetRsCb, StubRsGetRsCbV2, 10);
    mocker(calloc, 10, NULL);
    ret = RsTlvInit(phyId, &bufferSize);
    EXPECT_INT_EQ(-ENOMEM, ret);
    mocker_clean();

    mocker_invoke(RsGetRsCb, StubRsGetRsCbV2, 10);
    ret = RsTlvInit(phyId, &bufferSize);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
    FreeRsCb();
}

void TcRsNslbDeinit()
{
    unsigned int phyId = 0;
    int ret = 0;

    mocker_invoke(RsGetTlvCb, StubRsGetNslbCbDeinit, 10);
    ret = RsTlvDeinit(phyId);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker_invoke(RsGetTlvCb, StubRsGetNslbCbAfterDeinit, 10);
    ret = RsTlvDeinit(phyId);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsNslbRequest()
{
    struct TlvRequestMsgHead head = {0};
    unsigned int bufferSize = 0;
    char* dataOut = NULL;
    char* dataIn = NULL;
    int ret = 0;

    head.phyId = 0;
    head.type = 0;
    dataOut = (char*)calloc(16, sizeof(char));
    dataIn = (char*)calloc(16, sizeof(char));

    mocker_invoke(RsGetTlvCb, StubRsGetNslbCb, 10);
    mocker(RsTlvAssembleSendData, 10, -EINVAL);
    ret = RsTlvRequest(&head, dataIn, dataOut, &bufferSize, 3072);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
    FreeRsCb();

    head.offset = 0;
    mocker_invoke(RsGetTlvCb, StubRsGetNslbCb, 10);
    mocker_invoke(RsTlvAssembleSendData, StubRsTlvAssembleSendData, 10);
    ret = RsTlvRequest(&head, dataIn, dataOut, &bufferSize, 3072);
    EXPECT_INT_EQ(0, ret);
    FreeRsCb();

    head.offset = 1U;
    head.totalBytes = 16U;
    ret = RsTlvRequest(&head, dataIn, dataOut, &bufferSize, 3072);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
    FreeRsCb();

    free(dataIn);
    dataIn = NULL;
    free(dataOut);
    dataOut = NULL;
}

void TcRsTlvAssembleSendData()
{
    struct TlvRequestMsgHead head = {0};
    struct TlvBufInfo bufInfo = {0};
    bool sendFinish = false;
    char data[16] = {0};
    int ret = 0;

    bufInfo.bufferSize = RS_TLV_BUFFER_SIZE;
    bufInfo.buf = (char*)calloc(RS_TLV_BUFFER_SIZE, sizeof(char));
    memset_s(bufInfo.buf, bufInfo.bufferSize, 0, bufInfo.bufferSize);
    head.sendBytes = 16U;
    head.totalBytes = 16U;
    head.offset = 0;
    head.phyId = 0;
    head.type = 0;

    mocker(memset_s, 10, 0);
    mocker(memcpy_s, 10, 0);
    ret = RsTlvAssembleSendData(&bufInfo, &head, data, &sendFinish, 3072);
    EXPECT_INT_EQ(0, ret);

    head.offset = 0;
    head.sendBytes = 8U;
    head.totalBytes = 16U;
    ret = RsTlvAssembleSendData(&bufInfo, &head, data, &sendFinish, 3072);
    EXPECT_INT_EQ(0, ret);

    head.offset = 16U;
    head.sendBytes = 16U;
    head.totalBytes = 16U;
    ret = RsTlvAssembleSendData(&bufInfo, &head, data, &sendFinish, 3072);
    EXPECT_INT_NE(0, ret);

    head.offset = RS_TLV_BUFFER_SIZE;
    ret = RsTlvAssembleSendData(&bufInfo, &head, data, &sendFinish, 3072);
    EXPECT_INT_EQ(-EINVAL, ret);

    head.offset = 0;
    head.sendBytes = 3073U;
    ret = RsTlvAssembleSendData(&bufInfo, &head, data, &sendFinish, 3072);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    free(bufInfo.buf);
    bufInfo.buf = NULL;
}

void TcRsEpollNslbEventHandle()
{
    struct epoll_event testEvents = {0};
    struct rs_cb testRsCb = {0};
    int ret = 0;

    testRsCb.tlvCb.nslbCb.initFlag = true;
    testEvents.events = 0;

    mocker(RsEpollNslbEventHandle, 10, NET_CO_PROCED);
    RsEpollEventHandleOne(&testRsCb, &testEvents);
    mocker_clean();

    pthread_mutex_init(&testRsCb.tlvCb.nslbCb.mutex, NULL);
    ret = RsEpollNslbEventHandle(&testRsCb.tlvCb.nslbCb, 0, 0);
    EXPECT_INT_EQ(NET_CO_PROCED, ret);
    mocker_clean();

    mocker(RsNetcoEventDispatch, 10, -1);
    ret = RsEpollNslbEventHandle(&testRsCb.tlvCb.nslbCb, 0, 0);
    EXPECT_INT_NE(NET_CO_PROCED, ret);
    mocker_clean();
    pthread_mutex_destroy(&testRsCb.tlvCb.nslbCb.mutex);
}

void TcRsGetTlvCb()
{
    struct RsTlvCb* tlvCb = NULL;
    uint32_t phyId = 0;
    int ret = 0;

    mocker_invoke(RsGetRsCb, StubRsGetRsCbV2, 10);
    ret = RsGetTlvCb(phyId, &tlvCb);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsNslbApiInit()
{
    NetCoIpPortArg arg = {0};
    unsigned int dataLen = 0;
    unsigned int type = 0;
    void* stubCo = NULL;
    char* data = NULL;
    int ret = 0;

    RsNetcoInit(0, arg);

    mocker(RsNetcoTblApiInit, 10, -1);
    ret = RsNetcoApiInit();
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsCcuRequest()
{
    struct TlvRequestMsgHead head = {0};
    char dataOut[MAX_TLV_MSG_DATA_LEN_V2];
    char dataIn[MAX_TLV_MSG_DATA_LEN_V2];
    unsigned int bufferSize = 0;
    int ret = 0;

    head.type = MSG_TYPE_CCU_INIT;
    mocker(RsCcuInit, 10, -1);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsCcuInit, 10, 0);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    head.type = MSG_TYPE_CCU_UNINIT;
    mocker(RsCcuUninit, 10, -1);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsCcuUninit, 10, 0);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    head.type = MSG_TYPE_CCU_GET_MEM_INFO;
    mocker(RsCcuGetMemInfo, 10, -1);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    head.type = MSG_TYPE_CCU_DISPATCH_CMD;
    mocker(RsCcuCustomChannel, 10, -1);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsCcuCustomChannel, 10, 0);
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    head.type = MSG_TYPE_CCU_MAX;
    mocker(isCcuTlvReqExist, 10, false);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    head.type = MSG_TYPE_CCU_DISPATCH_CMD;
    mocker(isCcuTlvReqExist, 10, true);
    mocker(RsCcuTlvRequest, 10, -1);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(isCcuTlvReqExist, 10, true);
    mocker(RsCcuTlvRequest, 10, 0);
    ret = RsCcuRequest(&head, dataIn, dataOut, &bufferSize);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void tc_RsNslbNetcoInitDeinit()
{
    struct RsNslbCb nslbCb = {0};
    char netcoCb = 0;
    int ret = 0;

    mocker(RsNetcoInitArg, 10, 0);
    mocker(RsNslbApiInit, 10, 0);
    mocker_invoke(RsGetRsCb, StubRsGetRsCbV2, 10);
    mocker(RsNetcoInit, 10, (int)(uintptr_t)&netcoCb);
    ret = RsNslbNetcoInit(0, &nslbCb);
    EXPECT_INT_EQ(0, ret);

    mocker(RsNetcoDeinit, 10, 0);
    mocker(RsNslbApiDeinit, 10, 0);
    RsNslbNetcoDeinit(&nslbCb);
    mocker_clean();
}

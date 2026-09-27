/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include "tc_adp.h"
#include <stdlib.h>
#include <sched.h>
#include <stdint.h>
#include "ut_dispatch.h"
#include "rs.h"
#include "rs_ping.h"
#include "rs_tlv.h"
#include "ra_adp.h"
#include "ra_adp_tlv.h"
#include "ra_adp_ctx.h"
#include "ra_hdc.h"
#include "ra_hdc_lite.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_socket.h"
#include "ra_hdc_tlv.h"
#include "ra_hdc_ctx.h"
#include "ra_hdc_async_ctx.h"
#include "errno.h"
#undef TOKEN_RATE
#define TOKEN_RATE
extern struct RsCtxOps gRaRsCtxOps;
extern int RecvHandleSendPkt(HDC_SESSION session, unsigned int* chipId);
static int counter = 0;
int StubRecvHandleSendPkt0(HDC_SESSION session, unsigned int* closeSession)
{
    counter++;
    if (counter <= 1) {
        *closeSession = 0;
        return 0;
    } else {
        *closeSession = 1;
        return 1;
    }
}

int StubRecvHandleSendPkt(HDC_SESSION session, unsigned int* closeSession)
{
    *closeSession = 1;
    return 0;
}

extern struct RaHdcInitPara gHdcInitPara;
static char* gTestMsg[MAX_TEST_MESSAGE] = {0};
static int gMsgCount = 0;
static int gCurrentMsgIndex = 0;
static int gAcceptTimes = 0;
static HDC_SESSION gTestSession;
static pid_t gHostTgid = 0;

static void WaitForHccpReady(void)
{
    int timeout;

    /* Phase 1: wait for Thread B to start AND consume at least one message.
     * threadStatus == RUNNING: set by RaPthread under mutex (ra_adp.c:1850)
     * gCurrentMsgIndex > 0: monotonically increasing progress, cannot be missed
     * Combined guarantee: session accepted, Thread B running, message processing started */
    timeout = 500;
    while ((gHdcInitPara.threadStatus != THREAD_RUNNING || gCurrentMsgIndex == 0) && --timeout > 0) {
        usleep(1000);
    }

    /* Phase 2: wait for CLOSE message to be fully processed.
     * connectStatus set back to HDC_UNCONNECTED after CLOSE handler completes (ra_adp.c:1812)
     * CLOSE is the last message (FIFO), so all prior messages are also done */
    timeout = 500;
    while (gHdcInitPara.connectStatus != HDC_UNCONNECTED && --timeout > 0) {
        usleep(1000);
    }
}

DLLEXPORT drvError_t StubGetMsgBuffer(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen)
{
    usleep(1000);
    *pBuf = gTestMsg[gCurrentMsgIndex++];
    struct MsgHead* tmsg = (struct MsgHead*)*pBuf;
    *pLen = tmsg->msgDataLen + sizeof(struct MsgHead);
    return 0;
}

char* AddTestMsg(unsigned int opcode, unsigned int msgLen)
{
#define MAX_HDC_DATA_SIZE (4096 - 256 + 64)
    char* temp = (char*)calloc(sizeof(char), sizeof(struct MsgHead) + msgLen);
    struct MsgHead* tmsg = (struct MsgHead*)temp;
    int ret;

    tmsg->opcode = opcode;
    tmsg->msgDataLen = msgLen;
    if (opcode != RA_RS_SEND_WRLIST_EXT && opcode != RA_RS_SEND_WRLIST && opcode != RA_RS_WLIST_DEL
        && opcode != RA_RS_WLIST_ADD && opcode != RA_RS_GET_VNIC_IP_INFOS_V1) {
        ret = (msgLen > MAX_HDC_DATA_SIZE) ? 1 : 0;
        if (ret != 0) {
            printf("%s: opcode:%u, msg_len:%u exceeds %u\n", __func__, opcode, msgLen, MAX_HDC_DATA_SIZE);
        }
        EXPECT_INT_EQ(ret, 0);
    }

    tmsg->hostTgid = gHostTgid;
    gTestMsg[gMsgCount++] = temp;
    return temp;
}

DLLEXPORT drvError_t StubAcceptSession(HDC_SERVER server, HDC_SESSION* session)
{
    while (gAcceptTimes > 0) {
        *session = &gTestSession;
        --gAcceptTimes;
        return 0;
    }
    return -1;
}

void MsgClear()
{
    int i;
    for (i = 0; i < gMsgCount; ++i) {
        free(gTestMsg[i]);
        gTestMsg[i] = NULL;
    }
    gMsgCount = 0;
    gCurrentMsgIndex = 0;
}

void TcAdpEnvInit()
{
    mocker_clean();
    MsgClear();
    mocker((stub_fn_t)halHdcRecv, 10, 0);
    mocker((stub_fn_t)halHdcSend, 10, 0);
    mocker_invoke((stub_fn_t)drvHdcGetMsgBuffer, (stub_fn_t)StubGetMsgBuffer, 10);
    mocker_invoke((stub_fn_t)drvHdcSessionAccept, (stub_fn_t)StubAcceptSession, 10);
    gAcceptTimes = 1;
}
void TcCommonTest()
{
    unsigned int devid = 0;
    AddTestMsg(RA_RS_HDC_SESSION_CLOSE, sizeof(union OpHdcCloseData));
    int ret = HccpInit(devid, gHostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(ret, 0);

    WaitForHccpReady();

    // Wait for T_accept (RaHwHdcInit) thread to fully exit before deinit.
    // hdcFlag lifecycle: 1=thread running (ra_adp.c:2034), 0=thread exited (ra_adp.c:2051).
    // Symmetric with RaHwInit waiting for hdcFlag==1 (ra_adp.c:2139-2142).
    int hdcFlagTimeout = 500;
    while (gHdcInitPara.hdcFlag != 0 && --hdcFlagTimeout > 0) {
        usleep(1000);
    }
    EXPECT_INT_EQ(gHdcInitPara.hdcFlag, 0);

    ret = HccpDeinit(devid);
    EXPECT_INT_EQ(ret, 0);

    MsgClear();
    mocker_clean();
}

void TcHccpInitFail()
{
    unsigned int devid = 0;
    pid_t hostTgid = 0;
    mocker_clean();
    mocker((stub_fn_t)sched_setaffinity, 1, -1);
    int ret = HccpInit(devid, hostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    mocker((stub_fn_t)sched_setaffinity, 10, 0);
    mocker((stub_fn_t)pthread_create, 1, -1);
    mocker((stub_fn_t)usleep, 6000, 0);
    ret = HccpInit(devid, hostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    ret = HccpDeinit(devid);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker((stub_fn_t)pthread_create, 10, 0);
    mocker((stub_fn_t)usleep, 6000, 0);
    ret = HccpInit(devid, hostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    EXPECT_INT_NE(ret, 0);
    ret = HccpDeinit(devid);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    mocker((stub_fn_t)pthread_detach, 1, -1);
    mocker((stub_fn_t)RsInit, 1, -1);
    AddTestMsg(RA_RS_HDC_SESSION_CLOSE, sizeof(union OpSocketCloseData));
    ret = HccpInit(devid, hostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(gCurrentMsgIndex, 0);
    EXPECT_INT_NE(ret, 0);
    ret = HccpDeinit(devid);
    EXPECT_INT_EQ(ret, 0);

    mocker_clean();
    MsgClear();
    AddTestMsg(RA_RS_HDC_SESSION_CLOSE, sizeof(union OpSocketCloseData));
    ret = HccpInit(RA_MAX_PHY_ID_NUM, hostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(gCurrentMsgIndex, 0);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    MsgClear();
    mocker((stub_fn_t)drvHdcServerCreate, 1, -1);
    AddTestMsg(RA_RS_HDC_SESSION_CLOSE, sizeof(union OpSocketCloseData));
    ret = HccpInit(devid, hostTgid, HDC_SERVICE_TYPE_RDMA, WHITE_LIST_ENABLE);
    EXPECT_INT_EQ(gCurrentMsgIndex, 0);
    EXPECT_INT_NE(ret, 0);

    mocker_clean();
    MsgClear();
}

void TcHccpDeinitFail()
{
    mocker_clean();
    unsigned int devid = 0;
    int ret = 0;
    mocker((stub_fn_t)RsDeinit, 1, -1);
    ret = HccpDeinit(devid);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
}

void TcHccpInit()
{
    TcAdpEnvInit();
    TcCommonTest();

    MsgClear();
    mocker_clean();
}

void TcSocketConnect()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketBatchConnect, 1, 0);
    AddTestMsg(RA_RS_SOCKET_CONN, sizeof(union OpSocketConnectData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketBatchConnect, 1, -1);
    AddTestMsg(RA_RS_SOCKET_CONN, sizeof(union OpSocketConnectData));
    TcCommonTest();
}

void TcSocketClose()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketBatchClose, 1, 0);
    AddTestMsg(RA_RS_SOCKET_CLOSE, sizeof(union OpSocketCloseData));
    TcCommonTest();
}

void TcSocketAbort()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketBatchAbort, 1, 0);
    AddTestMsg(RA_RS_SOCKET_ABORT, sizeof(union OpSocketConnectData));
    TcCommonTest();

    mocker_clean();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketBatchAbort, 1, -1);
    AddTestMsg(RA_RS_SOCKET_ABORT, sizeof(union OpSocketConnectData));
    TcCommonTest();
}

void TcSocketListenStart()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketListenStart, 1, 0);
    AddTestMsg(RA_RS_SOCKET_LISTEN_START, sizeof(union OpSocketListenData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketListenStart, 1, -1);
    AddTestMsg(RA_RS_SOCKET_LISTEN_START, sizeof(union OpSocketListenData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketListenStart, 1, -98);
    AddTestMsg(RA_RS_SOCKET_LISTEN_START, sizeof(union OpSocketListenData));
    TcCommonTest();
}

void TcSocketListenStop()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketListenStop, 1, 0);
    AddTestMsg(RA_RS_SOCKET_LISTEN_STOP, sizeof(union OpSocketListenData));
    TcCommonTest();
}

void TcSocketInfo()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetSockets, 1, 0);
    AddTestMsg(RA_RS_GET_SOCKET, sizeof(union OpSocketInfoData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetSockets, 1, -1);
    AddTestMsg(RA_RS_GET_SOCKET, sizeof(union OpSocketInfoData));
    TcCommonTest();
}

void TcSocketSend()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketSend, 1, 0);
    AddTestMsg(RA_RS_SOCKET_SEND, sizeof(union OpSocketSendData));
    TcCommonTest();
}

void TcSocketRecv()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketRecv, 1, 0);
    AddTestMsg(RA_RS_SOCKET_RECV, sizeof(union OpSocketRecvData));
    TcCommonTest();
}

void TcSocketInit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketInit, 1, 0);
    AddTestMsg(RA_RS_SOCKET_INIT, sizeof(union OpSocketInitData));
    TcCommonTest();
}

void TcSocketDeinit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketDeinit, 1, 0);
    AddTestMsg(RA_RS_SOCKET_DEINIT, sizeof(union OpSocketDeinitData));
    TcCommonTest();
}

void TcSetTsqpDepth()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSetTsqpDepth, 1, 0);
    AddTestMsg(RA_RS_SET_TSQP_DEPTH, sizeof(union OpSetTsqpDepthData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSetTsqpDepth, 1, -1);
    AddTestMsg(RA_RS_SET_TSQP_DEPTH, sizeof(union OpSetTsqpDepthData));
    TcCommonTest();
}

void TcGetTsqpDepth()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetTsqpDepth, 1, 0);
    AddTestMsg(RA_RS_GET_TSQP_DEPTH, sizeof(union OpGetTsqpDepthData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetTsqpDepth, 1, -1);
    AddTestMsg(RA_RS_GET_TSQP_DEPTH, sizeof(union OpGetTsqpDepthData));
    TcCommonTest();
}

void TcQpCreate()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpCreate, 1, 0);
    AddTestMsg(RA_RS_QP_CREATE, sizeof(union OpQpCreateData));
    mocker((stub_fn_t)RsQpCreateWithAttrs, 10, 0);
    AddTestMsg(RA_RS_QP_CREATE_WITH_ATTRS, sizeof(union OpQpCreateWithAttrsData));
    AddTestMsg(RA_RS_AI_QP_CREATE, sizeof(union OpAiQpCreateData));
    AddTestMsg(RA_RS_AI_QP_CREATE_WITH_ATTRS, sizeof(union OpAiQpCreateWithAttrsData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpCreate, 1, -1);
    AddTestMsg(RA_RS_QP_CREATE, sizeof(union OpQpCreateData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpCreateWithAttrs, 10, -1);
    AddTestMsg(RA_RS_QP_CREATE_WITH_ATTRS, sizeof(union OpQpCreateWithAttrsData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpCreate, 10, -1);
    AddTestMsg(RA_RS_TYPICAL_QP_CREATE, sizeof(union OpTypicalQpCreateData));
    TcCommonTest();
}

void TcQpDestroy()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpDestroy, 1, 0);
    AddTestMsg(RA_RS_QP_DESTROY, sizeof(union OpQpDestroyData));
    TcCommonTest();
}

void TcQpStatus()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetQpStatus, 1, 0);
    AddTestMsg(RA_RS_QP_STATUS, sizeof(union OpQpStatusData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetQpStatus, 1, -1);
    AddTestMsg(RA_RS_QP_STATUS, sizeof(union OpQpStatusData));
    TcCommonTest();
}

void TcQpInfo()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetQpStatus, 1, 0);
    AddTestMsg(RA_RS_QP_INFO, sizeof(union OpQpInfoData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetQpStatus, 1, -1);
    AddTestMsg(RA_RS_QP_INFO, sizeof(union OpQpInfoData));
    TcCommonTest();
}

void TcQpConnect()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpConnectAsync, 1, 0);
    AddTestMsg(RA_RS_QP_CONNECT, sizeof(union OpQpConnectData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsTypicalQpModify, 1, -1);
    AddTestMsg(RA_RS_TYPICAL_QP_MODIFY, sizeof(union OpTypicalQpModifyData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsQpBatchModify, 1, -1);
    AddTestMsg(RA_RS_QP_BATCH_MODIFY, sizeof(union OpQpBatchModifyData));
    TcCommonTest();
}

void TcMrReg()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsMrReg, 1, 0);
    AddTestMsg(RA_RS_MR_REG, sizeof(union OpMrRegData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsTypicalRegisterMrV1, 1, 0);
    AddTestMsg(RA_RS_TYPICAL_MR_REG_V1, sizeof(union OpTypicalMrRegData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsTypicalRegisterMr, 1, 0);
    AddTestMsg(RA_RS_TYPICAL_MR_REG, sizeof(union OpTypicalMrRegData));
    TcCommonTest();
}

void TcMrDreg()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsMrDereg, 1, 0);
    AddTestMsg(RA_RS_MR_DEREG, sizeof(union OpMrDeregData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsTypicalDeregisterMr, 1, 0);
    AddTestMsg(RA_RS_TYPICAL_MR_DEREG, sizeof(union OpTypicalMrDeregData));
    TcCommonTest();
}

void TcSendWr()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWr, 1, 0);
    AddTestMsg(RA_RS_SEND_WR, sizeof(union OpSendWrData));
    TcCommonTest();
}

extern int memcpy_s(void* dest, size_t destMax, const void* src, size_t count);
void TcSendWrlist()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, 0);
    mocker((stub_fn_t)memcpy_s, 1, 0);
    AddTestMsg(RA_RS_SEND_WRLIST, sizeof(union OpSendWrlistData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, -1);
    mocker((stub_fn_t)memcpy_s, 1, 0);
    AddTestMsg(RA_RS_SEND_WRLIST, sizeof(union OpSendWrlistData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, -ENOENT);
    mocker((stub_fn_t)memcpy_s, 1, 0);
    AddTestMsg(RA_RS_SEND_WRLIST, sizeof(union OpSendWrlistData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, 0);
    mocker((stub_fn_t)memcpy_s, 1, -1);
    AddTestMsg(RA_RS_SEND_WRLIST, sizeof(union OpSendWrlistData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, 0);
    mocker((stub_fn_t)memcpy_s, 1, -1);
    AddTestMsg(RA_RS_SEND_WRLIST_EXT, sizeof(union OpSendWrlistDataExt));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, 0);
    mocker((stub_fn_t)memcpy_s, 1, -1);
    AddTestMsg(RA_RS_SEND_WRLIST_V2, sizeof(union OpSendWrlistDataV2));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, 0);
    mocker((stub_fn_t)memcpy_s, 1, -1);
    AddTestMsg(RA_RS_SEND_WRLIST_EXT_V2, sizeof(union OpSendWrlistDataExtV2));
    TcCommonTest();
}

void TcRdevInit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsRdevInit, 1, 0);
    AddTestMsg(RA_RS_RDEV_INIT, sizeof(union OpRdevInitData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsRdevInit, 1, -1);
    AddTestMsg(RA_RS_RDEV_INIT, sizeof(union OpRdevInitData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsRdevInitWithBackup, 1, 0);
    AddTestMsg(RA_RS_RDEV_INIT_WITH_BACKUP, sizeof(union OpRdevInitWithBackupData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsRdevInitWithBackup, 1, -1);
    AddTestMsg(RA_RS_RDEV_INIT_WITH_BACKUP, sizeof(union OpRdevInitWithBackupData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsRdevGetPortStatus, 1, -1);
    AddTestMsg(RA_RS_RDEV_GET_PORT_STATUS, sizeof(union OpRdevGetPortStatusData));
    TcCommonTest();
}

void TcRdevDeinit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsRdevDeinit, 1, 0);
    AddTestMsg(RA_RS_RDEV_DEINIT, sizeof(union OpRdevDeinitData));
    TcCommonTest();
}
void TcGetNotifyBa()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetNotifyMrInfo, 1, 0);
    AddTestMsg(RA_RS_GET_NOTIFY_BA, sizeof(union OpGetNotifyBaData));
    TcCommonTest();
}
void TcSetPid()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSetHostPid, 1, 0);
    AddTestMsg(RA_RS_SET_PID, sizeof(union OpSetPidData));
    TcCommonTest();
}
void TcGetVnicIp()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetVnicIp, 1, 0);
    AddTestMsg(RA_RS_GET_VNIC_IP, sizeof(union OpGetVnicIpData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetVnicIp, 1, -1);
    AddTestMsg(RA_RS_GET_VNIC_IP, sizeof(union OpGetVnicIpData));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetVnicIpInfos, 1, -1);
    AddTestMsg(RA_RS_GET_VNIC_IP_INFOS_V1, sizeof(union OpGetVnicIpInfosDataV1));
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetVnicIpInfos, 1, -1);
    AddTestMsg(RA_RS_GET_VNIC_IP_INFOS, sizeof(union OpGetVnicIpInfosData));
    TcCommonTest();
}
void TcSocketWhiteListAdd()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketWhiteListAdd, 1, 0);
    AddTestMsg(RA_RS_WLIST_ADD, sizeof(union OpWlistData));
    TcCommonTest();

    mocker_clean();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketWhiteListAdd, 1, -1);
    AddTestMsg(RA_RS_WLIST_ADD, sizeof(union OpWlistData));
    TcCommonTest();
}
void TcSocketWhiteListDel()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketWhiteListDel, 1, 0);
    AddTestMsg(RA_RS_WLIST_DEL, sizeof(union OpWlistData));
    TcCommonTest();

    mocker_clean();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketWhiteListDel, 1, -1);
    AddTestMsg(RA_RS_WLIST_DEL, sizeof(union OpWlistData));
    TcCommonTest();
}
void TcGetIfaddrs()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetIfaddrs, 2, 0);
    char* databuf = AddTestMsg(RA_RS_GET_IFADDRS, sizeof(union OpIfaddrData));
    union OpIfaddrData* ifaddrData = (union OpIfaddrData*)(databuf + sizeof(struct MsgHead));
    ifaddrData->txData.num = 1;

    databuf = AddTestMsg(RA_RS_GET_IFADDRS, sizeof(union OpIfaddrData));
    ifaddrData = (union OpIfaddrData*)(databuf + sizeof(struct MsgHead));
    ifaddrData->txData.num = 1;
    TcCommonTest();
}

void TcGetIfaddrsV2()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetIfaddrsV2, 10, 0);
    char* databuf = AddTestMsg(RA_RS_GET_IFADDRS_V2, sizeof(union OpIfaddrDataV2));
    union OpIfaddrDataV2* ifaddrData = (union OpIfaddrDataV2*)(databuf + sizeof(struct MsgHead));
    databuf = AddTestMsg(RA_RS_GET_IFADDRS_V2, sizeof(union OpIfaddrDataV2));

    ifaddrData->txData.num = 1;
    databuf = AddTestMsg(RA_RS_GET_IFADDRS_V2, sizeof(union OpIfaddrDataV2));
    TcCommonTest();
}

void TcGetIfnum()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetIfnum, 2, 0);
    char* databuf = AddTestMsg(RA_RS_GET_IFNUM, sizeof(union OpIfnumData));
    union OpIfnumData* ifnumData = (union OpIfnumData*)(databuf + sizeof(struct MsgHead));

    databuf = AddTestMsg(RA_RS_GET_IFNUM, sizeof(union OpIfnumData));
    ifnumData = (union OpIfnumData*)(databuf + sizeof(struct MsgHead));
    ifnumData->txData.num = 1;
    TcCommonTest();
}

void TcGetInterfaceVersion()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetInterfaceVersion, 1, 0);

    char* databuf = AddTestMsg(RA_RS_GET_INTERFACE_VERSION, sizeof(union OpGetVersionData));
    union OpGetVersionData* versionInfo = (union OpGetVersionData*)(databuf + sizeof(struct MsgHead));

    versionInfo->txData.opcode = 0;
    TcCommonTest();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetInterfaceVersion, 1, -1);
    databuf = AddTestMsg(RA_RS_GET_INTERFACE_VERSION, sizeof(union OpGetVersionData));
    versionInfo = (union OpGetVersionData*)(databuf + sizeof(struct MsgHead));
    versionInfo->txData.opcode = 0;
    TcCommonTest();
}

extern int RaRsNotifyCfgSet(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsNotifyCfgGet(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsSendWrListV2(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsSendWrList(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsSendWrListExtV2(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
extern int RaRsSendWrListExt(char* inBuf, char* outBuf, int* outLen, int* opResult, int rcvBufLen);
void TcSetNotifyCfg()
{
    int result = 0;
    mocker_clean();
    unsigned int size = sizeof(union OpNotifyCfgSetData) + sizeof(struct MsgHead);
    char* inBuf = calloc(1, size);
    union OpNotifyCfgSetData setNotifyBaData = {0};
    memcpy(inBuf + sizeof(struct MsgHead), &setNotifyBaData, sizeof(union OpNotifyCfgSetData));
    RaRsNotifyCfgSet(inBuf, NULL, NULL, &result, 1);

    free(inBuf);
    inBuf = NULL;

    TcAdpEnvInit();
    mocker((stub_fn_t)RsNotifyCfgSet, 1, 0);
    AddTestMsg(RA_RS_NOTIFY_CFG_SET, sizeof(union OpNotifyCfgSetData));
    TcCommonTest();
}

void TcGetNotifyCfg()
{
    int result = 0;
    mocker_clean();
    unsigned int size = sizeof(union OpNotifyCfgGetData) + sizeof(struct MsgHead);
    char* inBuf = calloc(1, size);
    char* outBuf = calloc(1, size);
    union OpNotifyCfgGetData getNotifyBaData = {0};
    memcpy(inBuf + sizeof(struct MsgHead), &getNotifyBaData, sizeof(union OpNotifyCfgGetData));
    memcpy(outBuf + sizeof(struct MsgHead), &getNotifyBaData, sizeof(union OpNotifyCfgGetData));
    RaRsNotifyCfgGet(inBuf, outBuf, NULL, &result, 1);

    free(outBuf);
    outBuf = NULL;
    free(inBuf);
    inBuf = NULL;

    TcAdpEnvInit();
    mocker((stub_fn_t)RsNotifyCfgGet, 1, 0);
    AddTestMsg(RA_RS_NOTIFY_CFG_GET, sizeof(union OpNotifyCfgGetData));
    TcCommonTest();
}

void TcRaRsSendWrListV2()
{
    union OpSendWrlistDataV2 sendWrlist;

    sendWrlist.txData.phyId = 0;
    sendWrlist.txData.rdevIndex = 0;
    sendWrlist.txData.qpn = 0;
    sendWrlist.txData.sendNum = 1;
    sendWrlist.txData.wrlist[0].op = 0;
    sendWrlist.txData.wrlist[0].sendFlags = 0;
    sendWrlist.txData.wrlist[0].dstAddr = 0;
    sendWrlist.txData.wrlist[0].memList.addr = 0;
    sendWrlist.txData.wrlist[0].memList.len = 0;
    sendWrlist.txData.wrlist[0].memList.lkey = 0;

    union OpSendWrlistDataV2 sendWrlistOut;

    char* inBuf = (char*)(&sendWrlist);
    char* outBuf = (char*)(&sendWrlistOut);

    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 0;

    RaRsSendWrListV2(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
}

void TcRaRsSendWrList()
{
    union OpSendWrlistData sendWrlist;

    sendWrlist.txData.phyId = 0;
    sendWrlist.txData.rdevIndex = 0;
    sendWrlist.txData.qpn = 0;
    sendWrlist.txData.sendNum = 1;
    sendWrlist.txData.wrlist[0].op = 0;
    sendWrlist.txData.wrlist[0].sendFlags = 0;
    sendWrlist.txData.wrlist[0].dstAddr = 0;
    sendWrlist.txData.wrlist[0].memList.addr = 0;
    sendWrlist.txData.wrlist[0].memList.len = 0;
    sendWrlist.txData.wrlist[0].memList.lkey = 0;

    union OpSendWrlistData sendWrlistOut;

    char* inBuf = (char*)(&sendWrlist);
    char* outBuf = (char*)(&sendWrlistOut);

    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 0;

    RaRsSendWrList(inBuf, outBuf, &outLen, &opResult, rcvBufLen);

    TcRaRsSendWrListV2();
}

void TcRaRsSendWrListExtV2()
{
    union OpSendWrlistDataExtV2 sendWrlist;

    sendWrlist.txData.phyId = 0;
    sendWrlist.txData.rdevIndex = 0;
    sendWrlist.txData.qpn = 0;
    sendWrlist.txData.sendNum = 1;
    sendWrlist.txData.wrlist[0].op = 0;
    sendWrlist.txData.wrlist[0].sendFlags = 0;
    sendWrlist.txData.wrlist[0].dstAddr = 0;
    sendWrlist.txData.wrlist[0].memList.addr = 0;
    sendWrlist.txData.wrlist[0].memList.len = 0;
    sendWrlist.txData.wrlist[0].memList.lkey = 0;
    sendWrlist.txData.wrlist[0].aux.dataType = 0;
    sendWrlist.txData.wrlist[0].aux.reduceType = 0;
    sendWrlist.txData.wrlist[0].aux.notifyOffset = 0;

    union OpSendWrlistDataExtV2 sendWrlistOut;

    char* inBuf = (char*)(&sendWrlist);
    char* outBuf = (char*)(&sendWrlistOut);

    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 0;

    RaRsSendWrListExtV2(inBuf, outBuf, &outLen, &opResult, rcvBufLen);
}

void TcRaRsSendWrListExt()
{
    union OpSendWrlistDataExt sendWrlist;

    sendWrlist.txData.phyId = 0;
    sendWrlist.txData.rdevIndex = 0;
    sendWrlist.txData.qpn = 0;
    sendWrlist.txData.sendNum = 1;
    sendWrlist.txData.wrlist[0].op = 0;
    sendWrlist.txData.wrlist[0].sendFlags = 0;
    sendWrlist.txData.wrlist[0].dstAddr = 0;
    sendWrlist.txData.wrlist[0].memList.addr = 0;
    sendWrlist.txData.wrlist[0].memList.len = 0;
    sendWrlist.txData.wrlist[0].memList.lkey = 0;
    sendWrlist.txData.wrlist[0].aux.dataType = 0;
    sendWrlist.txData.wrlist[0].aux.reduceType = 0;
    sendWrlist.txData.wrlist[0].aux.notifyOffset = 0;

    union OpSendWrlistDataExt sendWrlistOut;

    char* inBuf = (char*)(&sendWrlist);
    char* outBuf = (char*)(&sendWrlistOut);

    int outLen = 0;
    int opResult = 0;
    int rcvBufLen = 0;

    RaRsSendWrListExt(inBuf, outBuf, &outLen, &opResult, rcvBufLen);

    TcRaRsSendWrListExtV2();
}

void TcRaRsSendNormalWrlist()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSendWrlist, 1, 0);
    AddTestMsg(RA_RS_SEND_NORMAL_WRLIST, sizeof(union OpSendNormalWrlistData));
    TcCommonTest();
}

void TcRaRsSetQpAttrQos()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSetQpAttrQos, 1, 0);
    AddTestMsg(RA_RS_SET_QP_ATTR_QOS, sizeof(union OpSetQpAttrQosData));
    TcCommonTest();
}

void TcRaRsSetQpAttrTimeout()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSetQpAttrTimeout, 1, 0);
    AddTestMsg(RA_RS_SET_QP_ATTR_TIMEOUT, sizeof(union OpSetQpAttrTimeoutData));
    TcCommonTest();
}

void TcRaRsSetQpAttrRetryCnt()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSetQpAttrRetryCnt, 1, 0);
    AddTestMsg(RA_RS_SET_QP_ATTR_RETRY_CNT, sizeof(union OpSetQpAttrRetryCntData));
    TcCommonTest();
}

void TcRaRsGetCqeErrInfo()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetCqeErrInfo, 1, 0);
    AddTestMsg(RA_RS_GET_CQE_ERR_INFO, sizeof(union OpGetCqeErrInfoData));
    TcCommonTest();
}

void TcRaRsGetCqeErrInfoNum()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetCqeErrInfoNum, 1, 0);
    AddTestMsg(RA_RS_GET_CQE_ERR_INFO_NUM, sizeof(union OpGetCqeErrInfoListData));
    TcCommonTest();
}

void TcRaRsGetCqeErrInfoList()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetCqeErrInfoList, 1, 0);
    AddTestMsg(RA_RS_GET_CQE_ERR_INFO_LIST, sizeof(union OpGetCqeErrInfoListData));
    TcCommonTest();
}

void TcRaRsGetLiteSupport()
{
    TcAdpEnvInit();
    AddTestMsg(RA_RS_GET_LITE_SUPPORT, sizeof(union OpLiteSupportData));
    TcCommonTest();
}

void tc_ra_RsGetLiteRdevCap()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetLiteRdevCap, 1, 0);
    AddTestMsg(RA_RS_GET_LITE_RDEV_CAP, sizeof(union OpLiteRdevCapData));
    TcCommonTest();
}

void TcRaRsGetLiteQpCqAttr()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetLiteQpCqAttr, 1, 0);
    AddTestMsg(RA_RS_GET_LITE_QP_CQ_ATTR, sizeof(union OpLiteQpCqAttrData));
    TcCommonTest();
}

void TcRaRsGetLiteConnectedInfo()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetLiteConnectedInfo, 1, 0);
    AddTestMsg(RA_RS_GET_LITE_CONNECTED_INFO, sizeof(union OpLiteConnectedInfoData));
    TcCommonTest();
}

void TcRaRsSocketWhiteListV2()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketWhiteListAdd, 1, 0);
    mocker((stub_fn_t)RsSocketWhiteListDel, 1, 0);
    AddTestMsg(RA_RS_WLIST_ADD_V2, sizeof(union OpWlistDataV2));
    AddTestMsg(RA_RS_WLIST_DEL_V2, sizeof(union OpWlistDataV2));
    TcCommonTest();
}

void TcRaRsSocketCreditAdd()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsSocketAcceptCreditAdd, 1, 0);
    AddTestMsg(RA_RS_ACCEPT_CREDIT_ADD, sizeof(union OpAcceptCreditData));
    TcCommonTest();
}

void TcRaRsGetLiteMemAttr()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetLiteMemAttr, 1, 0);
    AddTestMsg(RA_RS_GET_LITE_MEM_ATTR, sizeof(union OpLiteMemAttrData));
    TcCommonTest();
}

void TcRaRsPingInit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingInit, 1, 0);
    AddTestMsg(RA_RS_PING_INIT, sizeof(union OpPingInitData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingInit, 1, 12);
    AddTestMsg(RA_RS_PING_INIT, sizeof(union OpPingInitData));
    TcCommonTest();
}

void TcRaRsPingTargetAdd()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTargetAdd, 1, 0);
    AddTestMsg(RA_RS_PING_ADD, sizeof(union OpPingAddData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTargetAdd, 1, 12);
    AddTestMsg(RA_RS_PING_ADD, sizeof(union OpPingAddData));
    TcCommonTest();
}

void TcRaRsPingTaskStart()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTaskStart, 1, 0);
    AddTestMsg(RA_RS_PING_START, sizeof(union OpPingStartData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTaskStart, 1, 12);
    AddTestMsg(RA_RS_PING_START, sizeof(union OpPingStartData));
    TcCommonTest();
}

void TcRaRsPingGetResults()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingGetResults, 1, 0);
    AddTestMsg(RA_RS_PING_GET_RESULTS, sizeof(union OpPingResultsData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingGetResults, 1, 12);
    AddTestMsg(RA_RS_PING_GET_RESULTS, sizeof(union OpPingResultsData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingGetResults, 1, -11);
    AddTestMsg(RA_RS_PING_GET_RESULTS, sizeof(union OpPingResultsData));
    TcCommonTest();
}

void TcRaRsPingTaskStop()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTaskStop, 1, 0);
    AddTestMsg(RA_RS_PING_STOP, sizeof(union OpPingStopData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTaskStop, 1, 12);
    AddTestMsg(RA_RS_PING_STOP, sizeof(union OpPingStopData));
    TcCommonTest();
}

void TcRaRsPingTargetDel()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTargetDel, 1, 0);
    AddTestMsg(RA_RS_PING_DEL, sizeof(union OpPingDelData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingTargetDel, 1, 12);
    AddTestMsg(RA_RS_PING_DEL, sizeof(union OpPingDelData));
    TcCommonTest();
}

void TcRaRsPingDeinit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingDeinit, 1, 0);
    AddTestMsg(RA_RS_PING_DEINIT, sizeof(union OpPingDeinitData));
    TcCommonTest();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsPingDeinit, 1, 12);
    AddTestMsg(RA_RS_PING_DEINIT, sizeof(union OpPingDeinitData));
    TcCommonTest();
}

void TcTlvInit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsTlvInit, 1, 0);
    AddTestMsg(RA_RS_TLV_INIT, sizeof(union OpTlvInitData));
    TcCommonTest();
}

void TcTlvDeinit()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RsTlvDeinit, 1, 0);
    AddTestMsg(RA_RS_TLV_DEINIT, sizeof(union OpTlvDeinitData));
    TcCommonTest();
}

void TcTlvRequest()
{
    TcAdpEnvInit();
    mocker((stub_fn_t)RaRsTlvRequest, 1, 0);
    AddTestMsg(RA_RS_TLV_REQUEST, sizeof(union OpTlvRequestData));
    TcCommonTest();
}

void TcRaRsRemapMr()
{
    mocker_clean();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsRemapMr, 1, 0);
    AddTestMsg(RA_RS_REMAP_MR, sizeof(union OpRemapMrData));
    TcCommonTest();
    mocker_clean();

    TcAdpEnvInit();
    mocker((stub_fn_t)RsRemapMr, 1, 1);
    AddTestMsg(RA_RS_REMAP_MR, sizeof(union OpRemapMrData));
    TcCommonTest();
    mocker_clean();
}

#define TC_ADP_CTX_OPS_TEST(stubFn, opcode, dataType) \
    do {                                              \
        TcAdpEnvInit();                               \
        mocker((stub_fn_t)stubFn, 1, 0);              \
        AddTestMsg(opcode, sizeof(dataType));         \
        TcCommonTest();                               \
    } while (0)

void TcRaRsTestCtxOps()
{
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.getDevEidInfoNum, RA_RS_GET_DEV_EID_INFO_NUM, union OpGetDevEidInfoNumData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.getDevEidInfoList, RA_RS_GET_DEV_EID_INFO_LIST, union OpGetDevEidInfoListData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxInit, RA_RS_CTX_INIT, union OpCtxInitData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxDeinit, RA_RS_CTX_DEINIT, union OpCtxDeinitData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxLmemReg, RA_RS_LMEM_REG, union OpLmemRegInfoData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxLmemUnreg, RA_RS_LMEM_UNREG, union OpLmemUnregInfoData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxRmemImport, RA_RS_RMEM_IMPORT, union OpRmemImportInfoData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxRmemUnimport, RA_RS_RMEM_UNIMPORT, union OpRmemUnimportInfoData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxChanCreate, RA_RS_CTX_CHAN_CREATE, union OpCtxChanCreateData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxChanDestroy, RA_RS_CTX_CHAN_DESTROY, union OpCtxChanDestroyData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxCqCreate, RA_RS_CTX_CQ_CREATE, union OpCtxCqCreateData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxCqDestroy, RA_RS_CTX_CQ_DESTROY, union OpCtxCqDestroyData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpCreate, RA_RS_CTX_QP_CREATE, union OpCtxQpCreateData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpDestroy, RA_RS_CTX_QP_DESTROY, union OpCtxQpDestroyData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpImport, RA_RS_CTX_QP_IMPORT, union OpCtxQpImportData);

    // ctxQpUnimport 需要两条消息
    TcAdpEnvInit();
    mocker((stub_fn_t)gRaRsCtxOps.ctxQpUnimport, 2, 0);
    AddTestMsg(RA_RS_CTX_QP_UNIMPORT_DEPRECATED, sizeof(union OpCtxQpUnimportData));
    AddTestMsg(RA_RS_CTX_QP_UNIMPORT, sizeof(union OpCtxQpUnimportData));
    TcCommonTest();

    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpBind, RA_RS_CTX_QP_BIND, union OpCtxQpBindData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpUnbind, RA_RS_CTX_QP_UNBIND, union OpCtxQpUnbindData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxBatchSendWr, RA_RS_CTX_BATCH_SEND_WR, union OpCtxBatchSendWrData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxUpdateCi, RA_RS_CTX_UPDATE_CI, union OpCtxUpdateCiData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxTokenIdAlloc, RA_RS_CTX_TOKEN_ID_ALLOC, union OpTokenIdAllocData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxTokenIdFree, RA_RS_CTX_TOKEN_ID_FREE, union OpTokenIdFreeData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.getTpInfoList, RA_RS_GET_TP_INFO_LIST, union OpGetTpInfoListData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpDestroyBatch, RA_RS_CTX_QP_DESTROY_BATCH, union OpCtxQpDestroyBatchData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxQpQueryBatch, RA_RS_CTX_QUERY_QP_BATCH, union OpCtxQpQueryBatchData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.getEidByIp, RA_RS_GET_EID_BY_IP, union OpGetEidByIpData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.getIpByEid, RA_RS_GET_IP_BY_EID, union OpGetIpByEidData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxGetAuxInfo, RA_RS_CTX_GET_AUX_INFO, union OpCtxGetAuxInfoData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.getTpAttr, RA_RS_GET_TP_ATTR, union OpGetTpAttrData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.setTpAttr, RA_RS_SET_TP_ATTR, union OpSetTpAttrData);
    TC_ADP_CTX_OPS_TEST(
        gRaRsCtxOps.ctxGetCrErrInfoList, RA_RS_CTX_GET_CR_ERR_INFO_LIST, union OpCtxGetCrErrInfoListData);
    TC_ADP_CTX_OPS_TEST(gRaRsCtxOps.ctxNotifyEvent, RA_RS_CTX_NOTIFY_EVENT, union OpCtxNotifyEventData);

    mocker_clean();
}

#undef TC_ADP_CTX_OPS_TEST

void TcRaRsGetTlsEnable0()
{
    mocker_clean();
    TcAdpEnvInit();
    mocker((stub_fn_t)RsGetTlsEnable, 1, 0);
    AddTestMsg(RA_RS_GET_TLS_ENABLE, sizeof(union OpGetTlsEnableData));
    TcCommonTest();
    mocker_clean();
}

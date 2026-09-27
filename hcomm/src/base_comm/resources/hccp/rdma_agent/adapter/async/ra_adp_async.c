/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sys/prctl.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "ra_hdc_async.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"
#include "ra_adp.h"
#include "ra_adp_pool.h"
#include "ra_adp_async.h"

struct RaHdcAsyncInfo gHdcAsync[RA_MAX_PHY_ID_NUM] = {0};
struct RaHdcInitPara gHdcAsyncInitPara = {0};
struct RsPthreadInfo gRaAsyncThreadInfo = {0};

int RaHwAsyncInit(unsigned int chipId, pid_t pid)
{
    int ret;

    ret = pthread_mutex_init(&gHdcAsyncInitPara.mutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("g_hdc_async_init_para mutex_init failed ret %d", ret), -ESYSFUNC);

    gHdcAsyncInitPara.chipId = chipId;
    gHdcAsyncInitPara.hostTgid = pid;

    ret = pthread_mutex_init(&gHdcAsync[chipId].sendMutex, NULL);
    if (ret != 0) {
        hccp_err("send_mutex mutex_init failed ret %d", ret);
        pthread_mutex_destroy(&gHdcAsyncInitPara.mutex);
        return -ESYSFUNC;
    }
    RaHdcInitOpSec(&gHdcAsync[chipId].opSec, BUCKET_DEPTH, true);
    return 0;
}

STATIC int RaHdcHandleSendPkt(unsigned int chipId, void *recvBuf, unsigned int recvLen)
{
    unsigned int closeSession = 0;
    void *sendBuf = NULL;
    int sendLen = 0;
    int ret;

    RsSetCtx(chipId);

    ret = RaHandle(&gHdcAsync[chipId].opSec, recvBuf, recvLen, (char **)&sendBuf, &sendLen, &closeSession);
    if (ret != 0) {
        hccp_err("ra_handle failed, ret:%d", ret);
        goto out;
    }

    ret = RaHdcAsyncSendPkt(&gHdcAsync[chipId], chipId, sendBuf, sendLen);
    if (ret != 0) {
        hccp_err("send_pkt failed, ret:%d", ret);
        goto err;
    }

err:
    free(sendBuf);
    sendBuf = NULL;
out:
    return ret;
}

STATIC void RaAsyncHandlePkt(unsigned int chipId, void *recvBuf, unsigned int recvLen)
{
    struct MsgHead *recvMsgHead = (struct MsgHead *)recvBuf;
    bool closeSession = false;

    // should handle RA_RS_HDC_SESSION_CLOSE on recv thread
    if (recvLen < sizeof(struct MsgHead) || recvMsgHead->opcode == RA_RS_HDC_SESSION_CLOSE) {
        closeSession = true;
    }
    if (closeSession) {
        (void)RaHdcHandleSendPkt(chipId, recvBuf, recvLen);
        RA_PTHREAD_MUTEX_LOCK(&gHdcAsyncInitPara.mutex);
        gHdcAsyncInitPara.connectStatus = HDC_UNCONNECTED;
        RA_PTHREAD_MUTEX_UNLOCK(&gHdcAsyncInitPara.mutex);
        return;
    }

    // handle other opcode: generate task and process the msg with work thread
    RaHdcPoolAddTask(gHdcAsync[chipId].pool, RaHdcHandleSendPkt, chipId, recvBuf, recvLen);
}

STATIC void *RaAsyncPthread(void *arg)
{
    (void)arg;
    unsigned int chipId = gHdcAsyncInitPara.chipId;
    unsigned int recvLen = 0;
    void *recvBuf = NULL;
    int ret;

    ret = pthread_detach(pthread_self());
    CHK_PRT_RETURN(ret != 0, hccp_err("pthread detach failed ret %d", ret), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_ra_async", 0, 0, 0);

    RA_PTHREAD_MUTEX_LOCK(&gHdcAsyncInitPara.mutex);
    gHdcAsyncInitPara.threadStatus = THREAD_RUNNING;
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcAsyncInitPara.mutex);

    RsGetCurTime(&gRaAsyncThreadInfo.lastCheckTime);
    ret = strncpy_s((char *)gRaAsyncThreadInfo.pthreadName, sizeof(gRaAsyncThreadInfo.pthreadName), "ra_async_thread",
        strlen("ra_async_thread"));
    CHK_PRT_RETURN(ret != 0, hccp_err("strncpy_s pthread name failed, ret[%d]", ret), NULL);

    hccp_run_info("pthread[%s] is alive!", gRaAsyncThreadInfo.pthreadName);
    while (1) {
        if (gHdcAsyncInitPara.threadStatus == THREAD_DESTROYING) {
            break;
        }

        if (gHdcAsyncInitPara.connectStatus != HDC_CONNECTED) {
            usleep(THREAD_SLEEP_TIME);
            continue;
        }
        RsHeartbeatAlivePrint(&gRaAsyncThreadInfo);
        // recv msg from hdc, alloc recv_buf in ra_async_pthread, free in work_pthread
        ret = RaHdcAsyncRecvPkt(&gHdcAsync[chipId], chipId, &recvBuf, &recvLen);
        if (ret != 0) {
            hccp_err("ra_hdc_async_recv_pkt failed, ret:%d chipId:%u", ret, chipId);
            break;
        }

        RaAsyncHandlePkt(chipId, recvBuf, recvLen);
    }

    hccp_info_others("thread [%d] is out, cleaning resources", getpid());
    RA_PTHREAD_MUTEX_LOCK(&gHdcAsyncInitPara.mutex);
    gHdcAsyncInitPara.threadStatus = THREAD_HALT;
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcAsyncInitPara.mutex);
    RA_PTHREAD_MUTEX_LOCK(&gHdcAsync[chipId].sendMutex);
    RaHdcCloseSession(&gHdcAsync[chipId].hdcSession);
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcAsync[chipId].sendMutex);
    return NULL;
}

STATIC void RaHwAsyncHdcInit(void *arg)
{
    (void)arg;
    unsigned int chipId = gHdcAsyncInitPara.chipId;
    pthread_t tidp;
    int ret;

    ret = pthread_detach(pthread_self());
    if (ret != 0) {
        hccp_err("pthread detach failed chip_id(%u), ret %d", chipId, ret);
        return;
    }

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_hw_async", 0, 0, 0);

    hccp_info_others("chip_id(%u)", chipId);
    gHdcAsyncInitPara.hdcFlag = 1;

    ret = pthread_create(&tidp, NULL, (void *)RaAsyncPthread, NULL);
    if (ret != 0) {
        hccp_err("Create pthread failed, chipId(%u), ret(%d) ", chipId, ret);
        return;
    }

    while (1) {
        if (gHdcAsyncInitPara.connectStatus != HDC_UNCONNECTED) {
            usleep(HDC_ACCEPT_SLEEP_TIME);
            continue;
        }
        ret = RaHdcSessionAccept(chipId, &gHdcAsync[chipId].hdcSession, (int)gHdcAsyncInitPara.hostTgid);
        if (ret != 0) {
            gHdcAsyncInitPara.hdcFlag = 0;
            return;
        }
        // should continue to accept: host_tgid != g_hdc_async_init_para.host_tgid
        if (ret == 0 && gHdcAsync[chipId].hdcSession == NULL) {
            continue;
        }

        RA_PTHREAD_MUTEX_LOCK(&gHdcAsyncInitPara.mutex);
        gHdcAsyncInitPara.connectStatus = HDC_CONNECTED;
        RA_PTHREAD_MUTEX_UNLOCK(&gHdcAsyncInitPara.mutex);
        return;
    }
}

void RaHwAsyncDeinit(void)
{
    pthread_mutex_destroy(&gHdcAsync[gHdcAsyncInitPara.chipId].sendMutex);
    pthread_mutex_destroy(&gHdcAsyncInitPara.mutex);
}

int RaRsAsyncHdcSessionConnect(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpAsyncHdcConnectData *asyncData = NULL;
    int timeout = RA_THREAD_TRY_TIME;
    unsigned int phyId = 0;
    pthread_t tidp;
    int ret;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpAsyncHdcConnectData), sizeof(struct MsgHead), rcvBufLen, opResult);
    asyncData = (union OpAsyncHdcConnectData *)(inBuf + sizeof(struct MsgHead));
    HCCP_CHECK_PARAM_LEN_RET_HOST(asyncData->txData.queueSize, 0, MAX_POOL_QUEUE_SIZE, opResult);
    HCCP_CHECK_PARAM_LEN_RET_HOST(asyncData->txData.threadNum, 0, MAX_POOL_THREAD_NUM, opResult);

    phyId = gHdcAsyncInitPara.chipId;
    gHdcAsync[phyId].pool = RaHdcPoolCreate(asyncData->txData.queueSize, asyncData->txData.threadNum);
    if (gHdcAsync[phyId].pool == NULL) {
        hccp_err("ra_hdc_pool_create failed, queueSize:%u threadNum:%u phyId:%u", asyncData->txData.queueSize,
            asyncData->txData.threadNum, asyncData->txData.phyId);
        *opResult = -ESYSFUNC;
        return 0;
    }

    ret = pthread_create(&tidp, NULL, (void *)RaHwAsyncHdcInit, NULL);
    if (ret != 0) {
        hccp_err("Create pthread failed, ret(%d)", ret);
        *opResult = -ESYSFUNC;
        RaHdcPoolDestroy(gHdcAsync[phyId].pool);
        gHdcAsync[phyId].pool = NULL;
        return 0;
    }

    // will block until time out: RA_THREAD_TRY_TIME * RA_THREAD_SLEEP_TIME us
    while (gHdcAsyncInitPara.hdcFlag != 1 && timeout > 0) {
        usleep(RA_THREAD_SLEEP_TIME);
        timeout--;
    }

    if (gHdcAsyncInitPara.hdcFlag == 0 || timeout <= 0) {
        hccp_err("HDC server thread create timeout, flag %d, timeout %d", gHdcAsyncInitPara.hdcFlag, timeout);
        *opResult = -ESRCH;
        RaHdcPoolDestroy(gHdcAsync[phyId].pool);
        gHdcAsync[phyId].pool = NULL;
        return 0;
    }

    *opResult = 0;
    return 0;
}

int RaRsAsyncHdcSessionClose(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)inBuf;
    (void)outBuf;
    (void)outLen;
    int tryAgain = HDC_TRY_TIME;
    unsigned int phyId = 0;

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpAsyncHdcCloseData), sizeof(struct MsgHead), rcvBufLen, opResult);

    RA_PTHREAD_MUTEX_LOCK(&gHdcAsyncInitPara.mutex);
    gHdcAsyncInitPara.threadStatus = THREAD_DESTROYING;
    RA_PTHREAD_MUTEX_UNLOCK(&gHdcAsyncInitPara.mutex);

    while ((gHdcAsyncInitPara.threadStatus != THREAD_HALT) && tryAgain != 0) {
        usleep(HDC_USLEEP_TIME);
        tryAgain--;
    }

    if (tryAgain <= 0) {
        hccp_warn_others("hdc async message thread quit timeout");
    }

    phyId = gHdcAsyncInitPara.chipId;
    RaHdcPoolDestroy(gHdcAsync[phyId].pool);
    gHdcAsync[phyId].pool = NULL;
    *opResult = 0;
    return 0;
}

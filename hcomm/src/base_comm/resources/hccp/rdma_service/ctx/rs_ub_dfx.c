/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <sys/types.h>
#include <urma_opcode.h>
#include <udma_u_ctl.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "dl_urma_function.h"
#include "ra_rs_err.h"
#include "rs_ctx_inner.h"
#include "rs_ub.h"
#include "rs_ub_dfx.h"

urma_cr_t gCrBuf[RS_WC_NUM];

STATIC int RsUbCtxGetCqeAuxInfo(struct RsUbDevCb *devCb, struct HccpAuxInfoIn *infoIn, struct HccpAuxInfoOut *infoOut)
{
    struct udma_u_cqe_aux_info_out cqeInfoOut = {0};
    struct udma_u_cqe_info_in cqeInfoIn = {0};
    urma_user_ctl_out_t out = {0};
    urma_user_ctl_in_t in = {0};
    int ret = 0;

    cqeInfoOut.aux_info_num = AUX_INFO_NUM_MAX;
    cqeInfoOut.aux_info_type = infoOut->auxInfoType;
    cqeInfoOut.aux_info_value = infoOut->auxInfoValue;
    cqeInfoIn.status = infoIn->cqe.status;
    cqeInfoIn.s_r = infoIn->cqe.sR;
    in.addr = (uint64_t)(uintptr_t)&cqeInfoIn;
    in.len = (uint32_t)sizeof(struct udma_u_cqe_info_in);
    in.opcode = UDMA_U_USER_CTL_QUERY_CQE_AUX_INFO;
    out.addr = (uint64_t)(uintptr_t)&cqeInfoOut;
    out.len = (uint32_t)sizeof(struct udma_u_cqe_aux_info_out);
    ret = RsUrmaUserCtl(devCb->urmaCtx, &in, &out);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_user_ctl query cqe aux info failed, ret:%d errno:%d", ret, errno),
        -EOPENSRC);

    infoOut->auxInfoNum = cqeInfoOut.aux_info_num;
    return ret;
}

STATIC int RsUbCtxGetAeAuxInfo(struct RsUbDevCb *devCb, struct HccpAuxInfoIn *infoIn, struct HccpAuxInfoOut *infoOut)
{
    struct udma_u_ae_aux_info_out aeInfoOut = {0};
    struct udma_u_ae_info_in aeInfoIn = {0};
    urma_user_ctl_out_t out = {0};
    urma_user_ctl_in_t in = {0};
    int ret = 0;

    aeInfoOut.aux_info_num = AUX_INFO_NUM_MAX;
    aeInfoOut.aux_info_type = infoOut->auxInfoType;
    aeInfoOut.aux_info_value = infoOut->auxInfoValue;
    aeInfoIn.event_type = infoIn->ae.eventType;
    in.addr = (uint64_t)(uintptr_t)&aeInfoIn;
    in.len = (uint32_t)sizeof(struct udma_u_ae_info_in);
    in.opcode = UDMA_U_USER_CTL_QUERY_AE_AUX_INFO;
    out.addr = (uint64_t)(uintptr_t)&aeInfoOut;
    out.len = (uint32_t)sizeof(struct udma_u_ae_aux_info_out);
    ret = RsUrmaUserCtl(devCb->urmaCtx, &in, &out);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_user_ctl query ae aux info failed, ret:%d errno:%d", ret, errno),
        -EOPENSRC);

    infoOut->auxInfoNum = aeInfoOut.aux_info_num;
    return ret;
}

int RsUbCtxGetAuxInfo(struct RsUbDevCb *devCb, struct HccpAuxInfoIn *infoIn, struct HccpAuxInfoOut *infoOut)
{
    int ret = 0;

    if (infoIn->type == AUX_INFO_IN_TYPE_CQE) {
        ret = RsUbCtxGetCqeAuxInfo(devCb, infoIn, infoOut);
    } else if (infoIn->type == AUX_INFO_IN_TYPE_AE) {
        ret = RsUbCtxGetAeAuxInfo(devCb, infoIn, infoOut);
    } else {
        hccp_err("invalid info_in->type[%d]", infoIn->type);
        ret = -EINVAL;
    }

    return ret;
}

STATIC void RsUdmaRetryTimeoutExceptionCheck(struct SensorNode *sensorNode, urma_cr_t *cr)
{
    int ret = 0;

    if (cr->status != URMA_CR_RNR_RETRY_CNT_EXC_ERR) {
        return;
    }

    ret = RsRetryTimeoutExceptionCheck(sensorNode);

    hccp_warn_rma("update sensor state logic_devid(%u), jettyId(%u), sensor_update_cnt(%d), ret(%d)\n",
        sensorNode->logicDevid, cr->local_id, sensorNode->sensorUpdateCnt, ret);
}

STATIC void RsUdmaSaveCqeErrInfo(uint32_t status, struct RsCtxJettyCb *jettyCb)
{
    RS_PTHREAD_MUTEX_LOCK(&jettyCb->crErrInfo.mutex);
    if (jettyCb->crErrInfo.info.status != 0) {
        RS_PTHREAD_MUTEX_ULOCK(&jettyCb->crErrInfo.mutex);
        return;
    }
    jettyCb->crErrInfo.info.status = status;
    jettyCb->crErrInfo.info.jettyId = jettyCb->jetty->jetty_id.id;
    RsGetCurTime(&jettyCb->crErrInfo.info.time);
    RS_PTHREAD_MUTEX_ULOCK(&jettyCb->crErrInfo.mutex);

    RS_PTHREAD_MUTEX_LOCK(&jettyCb->devCb->cqeErrCntMutex);
    jettyCb->devCb->cqeErrCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&jettyCb->devCb->cqeErrCntMutex);
}

STATIC void RsJfcCallbackProcess(struct RsCtxJettyCb *jettyCb, urma_cr_t *cr, urma_jfc_t *jfc)
{
    (void)jfc;
    if (cr->status != URMA_CR_SUCCESS) {
        RsUdmaSaveCqeErrInfo(cr->status, jettyCb);
        RsUdmaRetryTimeoutExceptionCheck(&jettyCb->devCb->rscb->sensorNode, cr);
    }
}

STATIC int RsHandleEpollPollJfc(struct RsUbDevCb *devCb, urma_jfce_t *jfce)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    urma_jfc_t *evJfc = NULL;
    uint32_t jettyId = 0;
    uint32_t ackCnt = 1;
    int polledCnt, i;
    int waitCnt = 0;
    int retTmp = 0;
    int ret = 0;

    waitCnt = RsUrmaWaitJfc(jfce, 1, 0, &evJfc);
    if (waitCnt == 0) {
        return -EAGAIN;
    }
    if (waitCnt != 1) {
        hccp_run_warn("rs_urma_wait_jfc failed, ret:%d errno:%d", waitCnt, errno);
        return -EOPENSRC;
    }
    RsUrmaAckJfc((urma_jfc_t **)&evJfc, &ackCnt, 1);

    polledCnt = RsUrmaPollJfc(evJfc, RS_WC_NUM, gCrBuf);
    if (polledCnt > RS_WC_NUM || polledCnt < 0) {
        hccp_run_warn("rs_urma_poll_jfc failed, ret:%d errno:%d", polledCnt, errno);
        ret = -EOPENSRC;
        goto rearm_jfc;
    }

    for (i = 0; i < polledCnt; i++) {
        jettyId = gCrBuf[i].local_id;
        RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
        ret = RsUbGetJettyCb(devCb, jettyId, &jettyCb);
        if (ret != 0) {
            hccp_err("get jetty_cb failed, ret:%d, jettyId[%u]:%u", ret, i, jettyId);
            RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
            break;
        }
        jettyCb->qpShareInfoAddr->ciVal += 1;
        RsJfcCallbackProcess(jettyCb, &(gCrBuf[i]), evJfc);
        RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
    }

rearm_jfc:
    retTmp = RsUrmaRearmJfc(evJfc, false);
    CHK_PRT_RETURN(retTmp != 0, hccp_err("rs_urma_rearm_jfc failed, retTmp:%d errno:%d", retTmp, errno), -EOPENSRC);
    return ret;
}

STATIC int RsHandleJfcEpollEvent(struct RsUbDevCb *devCb, int fd)
{
    struct RsCtxJfceCb *jfceCbCurr = NULL;
    struct RsCtxJfceCb *jfceCbNext = NULL;
    urma_jfce_t *jfceTmp = NULL;

    RS_LIST_GET_HEAD_ENTRY(jfceCbCurr, jfceCbNext, &devCb->jfceList, list, struct RsCtxJfceCb);
    for (; (&jfceCbCurr->list) != &devCb->jfceList;
         jfceCbCurr = jfceCbNext, jfceCbNext = list_entry(jfceCbNext->list.next, struct RsCtxJfceCb, list)) {
        jfceTmp = (urma_jfce_t *)(uintptr_t)jfceCbCurr->jfceAddr;
        if (jfceTmp->fd == fd) {
            return RsHandleEpollPollJfc(devCb, jfceTmp);
        }
    }

    return -ENODEV;
}

int RsEpollEventJfcInHandle(struct rs_cb *rsCb, int fd)
{
    struct RsUbDevCb *devCbCurr = NULL;
    struct RsUbDevCb *devCbNext = NULL;
    int ret = 0;

    if (rsCb->protocol != PROTOCOL_UDMA) {
        return -ENODEV;
    }

    RS_LIST_GET_HEAD_ENTRY(devCbCurr, devCbNext, &rsCb->udevList, list, struct RsUbDevCb);
    for (; (&devCbCurr->list) != &rsCb->udevList;
         devCbCurr = devCbNext, devCbNext = list_entry(devCbNext->list.next, struct RsUbDevCb, list)) {
        ret = RsHandleJfcEpollEvent(devCbCurr, fd);
        if (ret == -ENODEV) {
            continue;
        }
        return ret;
    }

    return -ENODEV;
}

STATIC int RsUbFillAsyncEventCb(urma_async_event_t *event, struct RsUbDevCb *devCb,
    struct RsCtxAsyncEventCb *asyncEventCb)
{
    int ret = 0;

    switch (event->event_type) {
        case URMA_EVENT_JFC_ERR:
            asyncEventCb->resId = event->element.jfc->jfc_id.id;
            ret = RsUrmaGetJfcOpt(event->element.jfc, URMA_JFC_FULL_CTX, asyncEventCb->context, JFC_CONTEXT_LEN);
            CHK_PRT_RETURN(ret != 0,
                hccp_err("RsUrmaGetJfcOpt failed, ret:%d, jfcId:%u devIndex:0x%0x", ret, event->element.jfc->jfc_id.id,
                    devCb->index),
                -EOPENSRC);
            asyncEventCb->len = JFC_CONTEXT_LEN;
            break;
        case URMA_EVENT_JFS_ERR:
            asyncEventCb->resId = event->element.jfs->jfs_id.id;
            break;
        case URMA_EVENT_JFR_ERR:
        case URMA_EVENT_JFR_LIMIT:
            asyncEventCb->resId = event->element.jfr->jfr_id.id;
            break;
        case URMA_EVENT_JETTY_ERR:
        case URMA_EVENT_JETTY_LIMIT:
            asyncEventCb->resId = event->element.jetty->jetty_id.id;
            ret = RsUrmaGetJettyOpt(event->element.jetty, URMA_JETTY_FULL_CTX, asyncEventCb->context,
                JETTY_CONTEXT_LEN);
            CHK_PRT_RETURN(ret != 0,
                hccp_err("RsUrmaGetJettyOpt failed, ret:%d, jettyId:%u devIndex:0x%0x", ret,
                    event->element.jetty->jetty_id.id, devCb->index),
                -EOPENSRC);
            asyncEventCb->len = JETTY_CONTEXT_LEN;
            break;
        case URMA_EVENT_JETTY_GRP_ERR:
            asyncEventCb->resId = event->element.jetty_grp->jetty_grp_id.id;
            break;
        case URMA_EVENT_PORT_ACTIVE:
        case URMA_EVENT_PORT_DOWN:
            asyncEventCb->resId = event->element.port_id;
            break;
        case URMA_EVENT_EID_CHANGE:
            asyncEventCb->resId = event->element.eid_idx;
            break;
        case URMA_EVENT_DEV_FATAL:
        case URMA_EVENT_ELR_ERR:
        case URMA_EVENT_ELR_DONE:
            asyncEventCb->resId = devCb->index;
            break;
        default:
            hccp_warn_rma("invalid event_type:%d devIndex:0x%x", event->event_type, devCb->index);
            break;
    }
    return ret;
}

STATIC int RsUbGetSaveAsyncEvent(struct RsUbDevCb *devCb)
{
    struct RsCtxAsyncEventCb *asyncEventCb = NULL;
    urma_async_event_t *event = NULL;
    int ret = 0;

    asyncEventCb = calloc(1, sizeof(struct RsCtxAsyncEventCb));
    CHK_PRT_RETURN(asyncEventCb == NULL, hccp_err("calloc asyncEventCb failed"), -ENOMEM);
    asyncEventCb->devCb = devCb;
    event = &asyncEventCb->asyncEvent;

    ret = RsUrmaGetAsyncEvent(devCb->urmaCtx, event);
    if (ret != 0) {
        hccp_err("RsUrmaGetAsyncEvent failed, ret:%d errno:%d devIndex:0x%x", ret, errno, devCb->index);
        ret = -EOPENSRC;
        goto free_event_cb;
    }
    RsUrmaAckAsyncEvent(event);

    ret = RsUbFillAsyncEventCb(event, devCb, asyncEventCb);
    if (ret != 0) {
        hccp_err("RsUbFillAsyncEventCb failed, ret:%d devIndex:0x%x", ret, devCb->index);
        goto free_event_cb;
    }

    hccp_run_info("get async event, event_type:%d resId:%d devIndex:0x%x", asyncEventCb->asyncEvent.event_type,
        asyncEventCb->resId, asyncEventCb->devCb->index);
    RsListAddTail(&asyncEventCb->list, &devCb->asyncEventList);
    devCb->asyncEventCnt++;

    return ret;

free_event_cb:
    free(asyncEventCb);
    asyncEventCb = NULL;
    return ret;
}

int RsEpollEventUrmaAsyncEventInHandle(struct rs_cb *rsCb, int fd)
{
    struct RsUbDevCb *devCbCurr = NULL;
    struct RsUbDevCb *devCbNext = NULL;
    int ret = 0;

    if (rsCb->protocol != PROTOCOL_UDMA) {
        return -ENODEV;
    }

    RS_LIST_GET_HEAD_ENTRY(devCbCurr, devCbNext, &rsCb->udevList, list, struct RsUbDevCb);
    for (; (&devCbCurr->list) != &rsCb->udevList;
         devCbCurr = devCbNext, devCbNext = list_entry(devCbNext->list.next, struct RsUbDevCb, list)) {
        RS_PTHREAD_MUTEX_LOCK(&devCbCurr->mutex);
        if (devCbCurr->urmaCtx != NULL && devCbCurr->urmaCtx->async_fd == fd) {
            ret = RsUbGetSaveAsyncEvent(devCbCurr);
            if (ret != 0) {
                hccp_err("rs_ub_get_save_async_event failed, ret:%d devIndex:0x%x", ret, devCbCurr->index);
            }
            RS_PTHREAD_MUTEX_ULOCK(&devCbCurr->mutex);
            return ret;
        }
        RS_PTHREAD_MUTEX_ULOCK(&devCbCurr->mutex);
    }

    return -ENODEV;
}

STATIC int RsUbCtxFillAsyncEvent(struct RsCtxAsyncEventCb *eventCb, struct AsyncEvent *asyncEvent)
{
    int ret = 0;

    asyncEvent->resId = eventCb->resId;
    asyncEvent->eventType = eventCb->asyncEvent.event_type;
    asyncEvent->len = eventCb->len;

    ret = memcpy_s(asyncEvent->context, CONTEXT_MAX_LEN, eventCb->context, eventCb->len);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("memcpy_s failed, ret:%d asyncEvent len:%u eventCb len:%u", ret, CONTEXT_MAX_LEN, eventCb->len),
        -ESAFEFUNC);

    return ret;
}

void RsUbCtxGetAsyncEvents(struct RsUbDevCb *devCb, struct AsyncEvent asyncEvents[], unsigned int *num)
{
    struct RsCtxAsyncEventCb *eventCbCurr = NULL;
    struct RsCtxAsyncEventCb *eventCbNext = NULL;
    unsigned int expectedNum = *num;
    int ret = 0;

    *num = 0;
    if (RsListEmpty(&devCb->asyncEventList)) {
        return;
    }

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RS_LIST_GET_HEAD_ENTRY(eventCbCurr, eventCbNext, &devCb->asyncEventList, list, struct RsCtxAsyncEventCb);
    for (; (&eventCbCurr->list) != &devCb->asyncEventList;
         eventCbCurr = eventCbNext, eventCbNext = list_entry(eventCbNext->list.next, struct RsCtxAsyncEventCb, list)) {
        ret = RsUbCtxFillAsyncEvent(eventCbCurr, &asyncEvents[*num]);
        if (ret != 0) {
            hccp_err("RsUbCtxFillAsyncEvent failed, ret:%d devIndex:0x%x", ret, devCb->index);
            RsUbFreeAsyncEventCb(devCb, eventCbCurr);
            break;
        }
        (*num)++;
        RsUbFreeAsyncEventCb(devCb, eventCbCurr);
        if (*num == expectedNum) {
            break;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

int RsUbGetJettyContext(struct RsUbDevCb *devCb, unsigned int id, uint8_t context[], unsigned int *len)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    int ret = 0;

    ret = RsUbGetJettyCb(devCb, id, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get jettyCb failed, ret:%d, jettyId:%u", ret, id), ret);

    ret = RsUrmaGetJettyOpt(jettyCb->jetty, URMA_JETTY_FULL_CTX, (void *)context, JETTY_CONTEXT_LEN);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsUrmaGetJettyOpt failed, ret:%d, jettyId:%u", ret, id), -EOPENSRC);

    *len = JETTY_CONTEXT_LEN;
    return ret;
}

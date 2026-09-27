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
#include <pthread.h>
#include "user_log.h"
#include "config_log.h"
#include "dl_hal_function.h"
#include "hccp_msg.h"
#include "hccp_common.h"
#include "ra_rs_err.h"
#include "rs_ub.h"
#include "rs_esched.h"

struct RsEschedInfo gRsEschedInfo = {0};

STATIC void RsEschedJettyDestroy(struct rs_cb *rscb, TsUbTaskReportT *taskInfo)
{
    unsigned int dieId, funcId, ueInfo;
    int ret, i;

    for (i = 0; i < taskInfo->num; i++) {
        dieId = taskInfo->array[i].udieId;
        funcId = taskInfo->array[i].functionId;
        ueInfo = RsGenerateUeInfo(dieId, funcId);
        ret = RsUbCtxJettyFree(rscb, ueInfo, taskInfo->array[i].jettyId);
        if (ret != 0) {
            hccp_run_warn("rs_ub_ctx_jetty_free unsuccessful, ret[%d] task_index[%d] logicId[%u] dieId[%u] "
                          "funcId[%u] jettyId[%u]",
                ret, i, rscb->logicId, dieId, funcId, taskInfo->array[i].jettyId);
            continue;
        }

        hccp_info_rma("jetty destroy task success, task_index[%d] logicId[%u] dieId[%u] funcId[%u] jettyId[%u]", i,
            rscb->logicId, dieId, funcId, taskInfo->array[i].jettyId);
    }
    return;
}

STATIC int RsEschedExecByCmdType(struct rs_cb *rscb, struct TagTsHccpMsg *msg)
{
    int ret = 0;

    switch (msg->cmdType) {
        case 0: // UB force kill
            RsEschedJettyDestroy(rscb, &msg->u.ubTaskInfo);
            break;
        default:
            hccp_run_warn("tag_ts_hccp_msg unsupported cmd type[%u]", msg->cmdType);
            ret = -EINVAL;
            break;
    }
    return ret;
}

STATIC void RsEschedCleanAllResource(struct rs_cb *rscb)
{
    struct RsUbDevCb *devCbCurr = NULL;
    struct RsUbDevCb *devCbNext = NULL;

    RS_PTHREAD_MUTEX_LOCK(&rscb->mutex);

    RS_LIST_GET_HEAD_ENTRY(devCbCurr, devCbNext, &rscb->udevList, list, struct RsUbDevCb);
    for (; (&devCbCurr->list) != &rscb->udevList;
         devCbCurr = devCbNext, devCbNext = list_entry(devCbNext->list.next, struct RsUbDevCb, list)) {
        hccp_info_rma("logicId[%u] devIndex[%u] start clean", rscb->logicId, devCbCurr->index);
        RsUbFreeJettyCbList(devCbCurr, &devCbCurr->jettyList, &devCbCurr->rjettyList);
    }

    RS_PTHREAD_MUTEX_ULOCK(&rscb->mutex);
    return;
}

STATIC int RsEschedProcessEvent(struct rs_cb *rscb, struct event_info *eventData)
{
    unsigned int subeventId = eventData->comm.subevent_id;
    struct TagTsHccpMsg *msg;
    uint16_t isAppExit;
    int ret = 0;

    CHK_PRT_RETURN(eventData->priv.msg_len != sizeof(struct TagTsHccpMsg),
        hccp_err("event invalid, msg_len[%u] != [%u], event_id[%d] subeventId[%u]", eventData->priv.msg_len,
            sizeof(struct TagTsHccpMsg), eventData->comm.event_id, subeventId),
        -EINVAL);

    msg = (struct TagTsHccpMsg *)eventData->priv.msg;
    isAppExit = msg->isAppExit;
    switch (isAppExit) {
        case 0: // host app alive, exec by cmd_type
            ret = RsEschedExecByCmdType(rscb, msg);
            break;
        case 1: // host app exit, clean all resource
            RsEschedCleanAllResource(rscb);
            break;
        default:
            hccp_run_warn("tag_ts_hccp_msg unsupported is_app_exit status[%u]", isAppExit);
            ret = -EINVAL;
            break;
    }

    return ret;
}

STATIC void RsEschedAckEvent(struct rs_cb *rscb, struct event_info *eventData)
{
    struct event_summary ackEvent = {0};
    int ret = 0;

    ackEvent.pid = eventData->comm.pid;
    ackEvent.grp_id = eventData->comm.grp_id;
    ackEvent.event_id = EVENT_HCCP_MSG;
    ackEvent.subevent_id = TOPIC_KILL_DONE_MSG;
    ackEvent.msg_len = eventData->priv.msg_len;
    ackEvent.msg = eventData->priv.msg;
    ackEvent.dst_engine = CCPU_DEVICE;
    ackEvent.policy = ONLY;
    ret = DlHalEschedSubmitEvent(rscb->logicId, &ackEvent);
    if (ret != 0) {
        hccp_run_warn("DlHalEschedSubmitEvent unsuccessful, ret[%d] logicId[%u]", ret, rscb->logicId);
    }

    return;
}

STATIC void RsEschedHandleEvent(struct rs_cb *rscb)
{
    struct event_info event = {0};
    int ret;

    ret = DlHalEschedWaitEvent(rscb->logicId, ESCHED_GRP_TS_HCCP, ESCHED_THREAD_ID_TS_HCCP, 0, &event);
    if (ret == -DRV_ERROR_SCHED_WAIT_TIMEOUT || ret == -DRV_ERROR_NO_EVENT) {
        return;
    }

    if (ret != DRV_ERROR_NONE) {
        hccp_run_warn("DlHalEschedWaitEvent unsuccessful, ret[%d] logicId[%u]", ret, rscb->logicId);
        return;
    }

    hccp_info_rma("wait event success, event_id[%d] subeventId[%u]", event.comm.event_id, event.comm.subevent_id);
    ret = RsEschedProcessEvent(rscb, &event);
    if (ret != 0) {
        hccp_run_warn("rs_esched_process_event unsuccessful, ret[%d] logicId[%u]", ret, rscb->logicId);
    }

    RsEschedAckEvent(rscb, &event);
}

STATIC void *RsEschedHandle(void *arg)
{
    struct rs_cb *rscb = (struct rs_cb *)arg;
    int ret;

    ret = pthread_detach(pthread_self());
    CHK_PRT_RETURN(ret, hccp_err("pthread detach failed ret %d", ret), NULL);
    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_rs_esched", 0, 0, 0);
    gRsEschedInfo.threadStatus = THREAD_RUNNING;

    while (1) {
        if (gRsEschedInfo.threadStatus == THREAD_DESTROYING) {
            break;
        }

        RsEschedHandleEvent(rscb);
        usleep(ESCHED_THREAD_USLEEP_TIME);
    }

    hccp_run_info("rs esched handle thread exit success, logic_devid[%u]", rscb->logicId);
    gRsEschedInfo.threadStatus = THREAD_HALT;
    return NULL;
}

int RsEschedInit(struct rs_cb *rscb)
{
    pthread_t rsEschedTid;
    int ret = 0;

    if (rscb->protocol != PROTOCOL_UDMA) {
        return 0;
    }

    ret = DlHalEschedAttachDevice(rscb->logicId);
    CHK_PRT_RETURN(ret != 0, hccp_err("halEschedSubscribeEvent failed, ret[%d] logicId[%u]", ret, rscb->logicId), ret);

    ret = DlHalEschedCreateGrp(rscb->logicId, ESCHED_GRP_TS_HCCP, GRP_TYPE_BIND_CP_CPU);
    CHK_PRT_RETURN(ret != 0, hccp_err("DlHalEschedCreateGrp failed, ret[%d] logicId[%u]", ret, rscb->logicId), ret);

    ret = DlHalEschedSubscribeEvent(rscb->logicId, ESCHED_GRP_TS_HCCP, ESCHED_THREAD_ID_TS_HCCP,
        (1UL << EVENT_HCCP_MSG));
    CHK_PRT_RETURN(ret != 0, hccp_err("DlHalEschedSubscribeEvent failed, ret[%d] logicId[%u]", ret, rscb->logicId),
        ret);

    ret = pthread_create(&rsEschedTid, NULL, RsEschedHandle, (void *)rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("pthread create failed, ret[%d] logicId[%u]", ret, rscb->logicId), -ESYSFUNC);

    return 0;
}

void RsEschedDeinit(enum ProtocolTypeT protocol)
{
    int tryAgain;

    if (protocol != PROTOCOL_UDMA) {
        return;
    }

    if (gRsEschedInfo.threadStatus == THREAD_HALT) {
        return;
    }

    // not need to use mutex, because rs_esched thread can only be created once when rs_init exec in hccp process
    gRsEschedInfo.threadStatus = THREAD_DESTROYING;

    tryAgain = ESCHED_THREAD_TRY_TIME;
    while ((gRsEschedInfo.threadStatus != THREAD_HALT) && tryAgain != 0) {
        usleep(ESCHED_THREAD_USLEEP_TIME);
        tryAgain--;
    }

    if (tryAgain <= 0) {
        hccp_warn_rma("rs_esched_handle thread quit timeout");
    }
    return;
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include <errno.h>
#include <sys/prctl.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "hccp_common.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "ra_rs_err.h"
#include "rs_inner.h"
#include "rs_epoll.h"
#include "rs_socket.h"
#include "rs_ping_inner.h"
#include "rs_ping_roce.h"
#include "rs_ping_urma.h"
#ifndef HNS_ROCE_LLT
#include "dlog_pub.h"
#endif
#include "rs_ping.h"

struct RsPthreadInfo gPingThreadInfo = {0};

int RsEpollEventPingHandle(struct rs_cb *rsCb, int fd)
{
    struct RsPingCtxCb *pingCb = &rsCb->pingCb;
    struct timeval timestamp2 = {0};
    int polledCnt = 0;
    int ret = -ENODEV;

    // thread not running, no need to handle ping
    if (pingCb->threadStatus != RS_PING_THREAD_RUNNING || pingCb->pingPongOps == NULL) {
        return ret;
    }

    // ping rq: receive detect packet
    if (pingCb->pingPongOps->checkPingFd(pingCb, fd)) {
        RS_PTHREAD_MUTEX_LOCK(&rsCb->pingCb.devMutex);
        if (pingCb->initCnt == 0) {
            goto free_dev_mutex;
        }
        ret = pingCb->pingPongOps->pingPollRcq(pingCb, &polledCnt, &timestamp2);
        if (ret != 0) {
            hccp_err("ping_poll_rcq failed, polledCnt:%d", polledCnt);
            goto free_dev_mutex;
        }
        pingCb->pingPongOps->pongHandleSend(pingCb, polledCnt, &timestamp2);
        goto free_dev_mutex;
    }

    // pong rq: receive response packet
    if (pingCb->pingPongOps->checkPongFd(pingCb, fd)) {
        RS_PTHREAD_MUTEX_LOCK(&rsCb->pingCb.devMutex);
        if (pingCb->initCnt == 0) {
            goto free_dev_mutex;
        }
        pingCb->pingPongOps->pongPollRcq(pingCb);
        ret = 0;
        goto free_dev_mutex;
    }

    return ret;

free_dev_mutex:
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->pingCb.devMutex);
    return ret;
}

STATIC void *RsPingHandle(void *arg)
{
    struct RsPingTargetInfo *targetNext = NULL;
    struct RsPingTargetInfo *targetCurr = NULL;
    struct rs_cb *rsCb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_NULL(arg);

    hccp_info_rma("<PING> thread begin! thread_id:%lu, pid:%d, ppid:%d", pthread_self(), getpid(), getppid());
    CHK_PRT_RETURN(pthread_detach(pthread_self()) != 0,
        hccp_err("pthread_detach failed! thread_id:%lu, errno:%d", pthread_self(), errno), NULL);

    (void)prctl(PR_SET_NAME, (uintptr_t) "hccp_ping", 0, 0, 0);

    rsCb = (struct rs_cb *)arg;

    RsGetCurTime(&gPingThreadInfo.lastCheckTime);
    ret = strncpy_s((char *)gPingThreadInfo.pthreadName, sizeof(gPingThreadInfo.pthreadName), "ping_pthread",
        strlen("ping_pthread"));
    CHK_PRT_RETURN(ret != 0, hccp_err("strncpy_s pthread name failed, ret[%d]", ret), NULL);

    hccp_run_info("pthread[%s] is alive!", gPingThreadInfo.pthreadName);
    while (1) {
        if (rsCb->pingCb.threadStatus != RS_PING_THREAD_RUNNING) {
            break;
        }

        RsHeartbeatAlivePrint(&gPingThreadInfo);
        if (rsCb->pingCb.taskStatus != RS_PING_TASK_RUNNING || rsCb->pingCb.taskAttr.packetCnt == 0) {
            usleep(RS_PING_PERIOD_TIME_USEC);
            continue;
        }
        if (RsListEmpty(&rsCb->pingCb.pingList)) {
            usleep(RS_PING_PERIOD_TIME_USEC);
            continue;
        }

        RS_LIST_GET_HEAD_ENTRY(targetCurr, targetNext, &rsCb->pingCb.pingList, list, struct RsPingTargetInfo);
        for (; rsCb->pingCb.taskStatus == RS_PING_TASK_RUNNING && (&targetCurr->list) != &rsCb->pingCb.pingList;
             targetCurr = targetNext, targetNext = list_entry(targetNext->list.next, struct RsPingTargetInfo, list)) {
            if (targetCurr->state != RS_PING_PONG_TARGET_READY) {
                usleep(rsCb->pingCb.taskAttr.packetInterval * RS_PING_MSEC_TO_USEC);
                continue;
            }

            ret = rsCb->pingCb.pingPongOps->pingPostSend(&rsCb->pingCb, targetCurr);
            if (ret != 0) {
                hccp_warn_rma("ping_post_send unsuccessful, ret:%d", ret);
                usleep(rsCb->pingCb.taskAttr.packetInterval * RS_PING_MSEC_TO_USEC);
                continue;
            }

            if (rsCb->pingCb.taskAttr.packetCnt == 1 && targetCurr->state == RS_PING_PONG_TARGET_READY) {
                targetCurr->state = RS_PING_PONG_TARGET_FINISH;
            }
            // make sure thread will exit
            usleep(rsCb->pingCb.taskAttr.packetInterval * RS_PING_MSEC_TO_USEC);

            // ping poll scq
            ret = rsCb->pingCb.pingPongOps->pingPollScq(&rsCb->pingCb, targetCurr);
            if (ret != 0) {
                continue;
            }
            targetCurr->resultSummary.sendCnt++;
        }

        // update task attr & status
        rsCb->pingCb.taskAttr.packetCnt--;
        if (rsCb->pingCb.taskAttr.packetCnt == 0) {
            rsCb->pingCb.taskStatus = RS_PING_TASK_RESET;
        }
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->pingCb.pingMutex);
    rsCb->pingCb.threadStatus = RS_PING_THREAD_FINISH;
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->pingCb.pingMutex);
    hccp_info_rma("<PING> QUIT thread_id:%lu, pid:%d", pthread_self(), getpid());
    return NULL;
}

STATIC int RsPingCbInitMutex(struct RsPingCtxCb *pingCb)
{
    int ret;

    ret = pthread_mutex_init(&pingCb->pingMutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init ping_mutex failed ret %d", ret);
        goto ping_mutex_init_failed;
    }
    ret = pthread_mutex_init(&pingCb->pongMutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init pong_mutex failed ret %d", ret);
        goto pong_mutex_init_failed;
    }
    ret = pthread_mutex_init(&pingCb->devMutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init dev_mutex failed ret %d", ret);
        goto dev_mutex_init_failed;
    }

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    RS_INIT_LIST_HEAD(&pingCb->pingList);
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongMutex);
    RS_INIT_LIST_HEAD(&pingCb->pongList);
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);

    return 0;

dev_mutex_init_failed:
    (void)pthread_mutex_destroy(&pingCb->pongMutex);
pong_mutex_init_failed:
    (void)pthread_mutex_destroy(&pingCb->pingMutex);
ping_mutex_init_failed:
    return -ESYSFUNC;
}

RS_ATTRI_VISI_DEF int RsPingHandleInit(unsigned int chipId, int hdcType, unsigned int whiteListStatus)
{
    struct rs_cb *rsCb = NULL;
    int ret;

    if (hdcType != HDC_SERVICE_TYPE_RDMA_V2 && whiteListStatus != WHITE_LIST_DISABLE) {
        return 0;
    }

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rs_cb failed, ret:%d, chipId:%u", ret, chipId), -ENODEV);

    ret = RsPingCbInitMutex(&rsCb->pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ping_cb_init_mutex failed, ret %d", ret), ret);

    rsCb->pingCb.threadStatus = RS_PING_THREAD_RUNNING;
    ret = pthread_create(&rsCb->pingCb.tid, NULL, (void *)RsPingHandle, (void *)rsCb);
    if (ret != 0) {
        hccp_err("Create pthread failed, ret(%d) ", ret);
        rsCb->pingCb.threadStatus = RS_PING_THREAD_RESET;
        (void)pthread_mutex_destroy(&rsCb->pingCb.pingMutex);
        (void)pthread_mutex_destroy(&rsCb->pingCb.pongMutex);
        (void)pthread_mutex_destroy(&rsCb->pingCb.devMutex);
        return -ESYSFUNC;
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsPingHandleDeinit(unsigned int chipId)
{
#define THREAD_STATUS_CHANGE_TIMEOUT 100
    struct rs_cb *rsCb = NULL;
    int ret;
    int i;

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rs_cb failed, ret:%d, chipId:%u", ret, chipId), -ENODEV);

    if (rsCb->pingCb.threadStatus != RS_PING_THREAD_RUNNING) {
        return 0;
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->pingCb.pingMutex);
    rsCb->pingCb.threadStatus = RS_PING_THREAD_RESET;
    rsCb->pingCb.taskStatus = RS_PING_TASK_RESET;
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->pingCb.pingMutex);

    // wait thread change to finish running status, wait 100 times(total cost: 1s) until timeout
    for (i = 0; i < THREAD_STATUS_CHANGE_TIMEOUT && rsCb->pingCb.threadStatus != RS_PING_THREAD_FINISH; i++) {
        usleep(RS_PING_PERIOD_TIME_USEC);
    }

    // thread not in finish running status, report timeout
    if (rsCb->pingCb.threadStatus != RS_PING_THREAD_FINISH) {
        hccp_run_info("<PING> wait thread tid:%lu finish running timeout, thread status:%d", rsCb->pingCb.tid,
            rsCb->pingCb.threadStatus);
    }

    (void)pthread_mutex_destroy(&rsCb->pingCb.pingMutex);
    (void)pthread_mutex_destroy(&rsCb->pingCb.pongMutex);
    (void)pthread_mutex_destroy(&rsCb->pingCb.devMutex);
    return 0;
}

STATIC int RsPingInitProtocolOps(struct RsPingCtxCb *pingCb, enum ProtocolTypeT protocol)
{
    pingCb->protocol = protocol;

    switch (protocol) {
        case PROTOCOL_RDMA:
            pingCb->pingPongOps = RsPingRoceGetOps();
            pingCb->pingPongDfx = RsPingRoceGetDfx();
            break;
        case PROTOCOL_UDMA:
            pingCb->pingPongOps = RsPingUrmaGetOps();
            pingCb->pingPongDfx = RsPingUrmaGetDfx();
            break;
        default:
            hccp_err("unsupported protocol:%u", protocol);
            return -EINVAL;
    }

    if (pingCb->pingPongOps == NULL || pingCb->pingPongOps->initPingCb == NULL || pingCb->pingPongDfx == NULL) {
        hccp_err("pingCb->pingPongOps or init_ping_cb or pingCb->ping_pong_dfx is NULL, protocol:%u", protocol);
        return -ENOTSUPP;
    }
    return 0;
}

RS_ATTRI_VISI_DEF int RsPingInit(struct PingInitAttr *attr, struct PingInitInfo *info, unsigned int *devIndex)
{
    struct RsPingCtxCb *pingCb = NULL;
    struct rs_cb *rscb = NULL;
    unsigned int phyId;
    int ret = 0;

    CHK_PRT_RETURN(attr == NULL || info == NULL || devIndex == NULL,
        hccp_err("param error, attr or info or devIndex is NULL"), -EINVAL);

    phyId = (attr->protocol == PROTOCOL_RDMA) ? attr->dev.rdma.phyId : attr->ub.phyId;
    ret = RsGetRsCb(phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsGetRsCb failed, phyId[%u] invalid, ret %d", phyId, ret), ret);

    pingCb = &rscb->pingCb;
    RS_PTHREAD_MUTEX_LOCK(&pingCb->devMutex);
    if (rscb->pingCb.initCnt != 0) {
        hccp_err("init_cnt:%u != 0", rscb->pingCb.initCnt);
        ret = -EEXIST;
        goto free_dev_mutex;
    }

    ret = rsGetLocalDevIDByHostDevID(phyId, &pingCb->logicDevid);
    if (ret != 0) {
        hccp_err("rsGetLocalDevIDByHostDevID failed, phyId(%u), ret(%d)", phyId, ret);
        goto free_dev_mutex;
    }

#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        // setup sharemem for pingmesh
        ret = RsSetupSharemem(rscb, false, phyId);
        if (ret != 0) {
            hccp_err("RsSetupSharemem failed, phyId(%u), ret(%d)", phyId, ret);
            goto free_dev_mutex;
        }
    }
#endif

    ret = RsPingInitProtocolOps(pingCb, attr->protocol);
    if (ret != 0) {
        hccp_err("rs_ping_init_protocol_ops failed, phyId:%u ret:%d", phyId, ret);
        goto free_dev_mutex;
    }

    ret = pingCb->pingPongOps->initPingCb(phyId, attr, info, devIndex, pingCb);
    if (ret != 0) {
        hccp_err("init_ping_cb failed, phyId:%u ret:%d", phyId, ret);
        goto free_dev_mutex;
    }

    pingCb->initCnt++;
    pingCb->pingPongDfx->initPingCbSuccess(phyId, attr, *devIndex);

free_dev_mutex:
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->devMutex);
    return ret;
}

STATIC int RsGetPingCb(struct RaRsDevInfo *rdev, struct RsPingCtxCb **pingCb)
{
    unsigned int phyId = rdev->phyId;
    struct rs_cb *rsCb = NULL;
    int ret;

    ret = RsGetRsCb(phyId, &rsCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsGetRsCb failed, phyId[%u] invalid, ret %d", phyId, ret), ret);

    CHK_PRT_RETURN(rdev->devIndex != rsCb->pingCb.devIndex,
        hccp_err("param error, devIndex:%u != pingCb.devIndex:%u", rdev->devIndex, rsCb->pingCb.devIndex), -ENODEV);

    CHK_PRT_RETURN(rsCb->pingCb.threadStatus != RS_PING_THREAD_RUNNING,
        hccp_err("thread_status:%d is not running", rsCb->pingCb.threadStatus), -ESRCH);

    *pingCb = &rsCb->pingCb;

    return 0;
}

RS_ATTRI_VISI_DEF int RsPingTargetAdd(struct RaRsDevInfo *rdev, struct PingTargetInfo *target)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    struct RsPingCtxCb *pingCb = NULL;
    int ret;

    CHK_PRT_RETURN(rdev == NULL || target == NULL, hccp_err("param error, rdev is NULL or target is NULL"), -EINVAL);
    CHK_PRT_RETURN(target->payload.size > PING_USER_PAYLOAD_MAX_SIZE,
        hccp_err("param error, size:%u > max_size:%u", target->payload.size, PING_USER_PAYLOAD_MAX_SIZE), -EINVAL);

    ret = RsGetPingCb(rdev, &pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_ping_cb failed, ret=%d phyId:%u", ret, rdev->phyId), ret);

    if (pingCb->taskStatus != RS_PING_TASK_RESET) {
        hccp_err("task_status:%d disallow to add target phyId:%u", pingCb->taskStatus, rdev->phyId);
        return -EEXIST;
    }

    ret = pingCb->pingPongOps->pingFindTargetNode(pingCb, &target->remoteInfo.qpInfo, &targetInfo);
    if (ret == 0) {
        hccp_info_rma("target node exist! phyId:%u", rdev->phyId);
        ret = -EEXIST;
        goto out;
    }

    ret = pingCb->pingPongOps->pingAllocTargetNode(pingCb, target, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_ping_alloc_target_node failed, ret:%d phyId:%u", ret, rdev->phyId);
        return ret;
    }

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    targetInfo->uuid = (uint64_t)pingCb->pingNum;
    RsListAddTail(&targetInfo->list, &pingCb->pingList);
    pingCb->pingNum++;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

out:
    pingCb->pingPongDfx->addTargetSuccess(target, targetInfo);
    return ret;
}

RS_ATTRI_VISI_DEF int RsPingTaskStart(struct RaRsDevInfo *rdev, struct PingTaskAttr *attr)
{
    struct RsPingTargetInfo *targetNext = NULL;
    struct RsPingTargetInfo *targetCurr = NULL;
    struct RsPingCtxCb *pingCb = NULL;
    unsigned int targetCnt = 0;
    int ret;

    CHK_PRT_RETURN(rdev == NULL || attr == NULL, hccp_err("param error, rdev is NULL or attr is NULL"), -EINVAL);
    ret = RsGetPingCb(rdev, &pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_ping_cb failed, ret=%d phyId:%u", ret, rdev->phyId), ret);

    if (pingCb->taskStatus != RS_PING_TASK_RESET) {
        hccp_warn_rma("task_status:%d disallow to start ping task, phyId:%u", pingCb->taskStatus, rdev->phyId);
        return -EEXIST;
    }
    CHK_PRT_RETURN(attr->packetCnt == 0 || attr->packetInterval == 0 || attr->timeoutInterval == 0,
        hccp_err("param error, packetCnt:%u or packetInterval:%u or timeoutInterval:%u is 0", attr->packetCnt,
            attr->packetInterval, attr->timeoutInterval),
        -EINVAL);

    pingCb->pingPongOps->resetRecvBuffer(pingCb);

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    pingCb->taskId++;
    (void)memcpy_s(&pingCb->taskAttr, sizeof(struct PingTaskAttr), attr, sizeof(struct PingTaskAttr));
    RS_LIST_GET_HEAD_ENTRY(targetCurr, targetNext, &pingCb->pingList, list, struct RsPingTargetInfo);
    for (; (&targetCurr->list) != &pingCb->pingList;
         targetCurr = targetNext, targetNext = list_entry(targetNext->list.next, struct RsPingTargetInfo, list)) {
        (void)memset_s(&targetCurr->resultSummary, sizeof(struct PingResultSummary), 0,
            sizeof(struct PingResultSummary));
        (void)memcpy_s(&targetCurr->resultSummary.taskAttr, sizeof(struct PingTaskAttr), attr,
            sizeof(struct PingTaskAttr));
        targetCurr->resultSummary.rttMin = ~0;
        targetCurr->resultSummary.taskId = pingCb->taskId;
        targetCurr->state = RS_PING_PONG_TARGET_READY;
        targetCnt++;
    }

    pingCb->taskStatus = RS_PING_TASK_RUNNING;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    hccp_info_rma("target_cnt:%u packet_cnt:%u packet_interval:%u timeout_interval:%u task_id:%u start success",
        targetCnt, attr->packetCnt, attr->packetInterval, attr->timeoutInterval, pingCb->taskId);
    return 0;
}

RS_ATTRI_VISI_DEF int RsPingGetResults(struct RaRsDevInfo *rdev, struct PingTargetCommInfo target[], unsigned int *num,
    struct PingResultInfo result[])
{
    struct RsPingCtxCb *pingCb = NULL;
    unsigned int expectedNum;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(rdev == NULL || num == NULL || target == NULL || result == NULL,
        hccp_err("param error, rdev is NULL or num is NULL or result/target is NULL"), -EINVAL);
    expectedNum = *num;
    *num = 0;
    ret = RsGetPingCb(rdev, &pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_ping_cb failed, ret=%d phyId:%u", ret, rdev->phyId), ret);

    // caller needs to retry, degrade log level
    if (pingCb->taskStatus == RS_PING_TASK_RUNNING) {
        hccp_warn_rma("task_status:%d disallow to get ping results phyId:%u", pingCb->taskStatus, rdev->phyId);
        return -EAGAIN;
    }

    for (i = 0; i < expectedNum; i++) {
        ret = pingCb->pingPongOps->getTargetResult(pingCb, &target[i], &result[i]);
        if (ret != 0) {
            hccp_err("rs_ping_get_target_result node i:%d failed phyId:%u", i, rdev->phyId);
            i = (i > 0) ? (i - 1U) : 0;
            goto out;
        }
    }

out:
    *num = i;
    return ret;
}

RS_ATTRI_VISI_DEF int RsPingTaskStop(struct RaRsDevInfo *rdev)
{
    struct RsPingCtxCb *pingCb = NULL;
    int ret;

    CHK_PRT_RETURN(rdev == NULL, hccp_err("param error, rdev is NULL"), -EINVAL);
    ret = RsGetPingCb(rdev, &pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_ping_cb failed, ret=%d phyId:%u", ret, rdev->phyId), ret);

    hccp_info_rma("task_status:%d modify to %d, phyId:%u", pingCb->taskStatus, RS_PING_TASK_RESET, rdev->phyId);

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    pingCb->taskStatus = RS_PING_TASK_RESET;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    return 0;
}

RS_ATTRI_VISI_DEF int RsPingTargetDel(struct RaRsDevInfo *rdev, struct PingTargetCommInfo target[], unsigned int *num)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    struct RsPingCtxCb *pingCb = NULL;
    unsigned int expectedNum;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(rdev == NULL || target == NULL || num == NULL,
        hccp_err("param error, rdev or target or num is NULL"), -EINVAL);
    ret = RsGetPingCb(rdev, &pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_ping_cb failed, ret=%d phyId:%u", ret, rdev->phyId), ret);

    if (pingCb->taskStatus != RS_PING_TASK_RESET) {
        hccp_err("task_status:%d disallow to delete target phyId:%u", pingCb->taskStatus, rdev->phyId);
        return -EEXIST;
    }

    expectedNum = *num;
    for (i = 0; i < expectedNum; i++) {
        ret = pingCb->pingPongOps->pingFindTargetNode(pingCb, &target[i].qpInfo, &targetInfo);
        if (ret != 0) {
            pingCb->pingPongDfx->pingCannotFindTargetNode(i, ret, target[i], rdev->phyId);
            goto out;
        }

        pingCb->pingPongOps->pingFreeTargetNode(pingCb, targetInfo);
        (void)pthread_mutex_destroy(&targetInfo->tripMutex);
        free(targetInfo);
        targetInfo = NULL;
    }

out:
    *num = i;
    return ret;
}

RS_ATTRI_VISI_DEF int RsPingDeinit(struct RaRsDevInfo *rdev)
{
    struct RsPingCtxCb *pingCb = NULL;
    int ret = 0;

    CHK_PRT_RETURN(rdev == NULL, hccp_err("param error, rdev is NULL"), -EINVAL);
    ret = RsGetPingCb(rdev, &pingCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_ping_cb failed, ret=%d phyId:%u", ret, rdev->phyId), ret);

    RS_PTHREAD_MUTEX_LOCK(&pingCb->devMutex);
    if (pingCb->initCnt == 0) {
        hccp_err("init_cnt is 0");
        ret = -ENODEV;
        goto free_dev_mutex;
    }

    pingCb->pingPongOps->deinitPingCb(rdev->phyId, pingCb);
    pingCb->initCnt--;
    hccp_run_info("pingCb deinit success, phyId:%u, devIndex:%u", rdev->phyId, rdev->devIndex);

free_dev_mutex:
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->devMutex);
    return ret;
}
